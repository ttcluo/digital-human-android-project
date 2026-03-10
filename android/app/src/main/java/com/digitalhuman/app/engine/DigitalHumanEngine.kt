package com.digitalhuman.app.engine

import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.util.Log
import java.io.File
import java.io.FileOutputStream

/**
 * 数字人推理引擎
 * Java/Kotlin接口
 */
class DigitalHumanEngine private constructor() {

    companion object {
        private const val TAG = "DigitalHumanEngine"
        
        @Volatile
        private var instance: DigitalHumanEngine? = null
        
        fun getInstance(): DigitalHumanEngine {
            return instance ?: synchronized(this) {
                instance ?: DigitalHumanEngine().also { instance = it }
            }
        }
    }
    
    private var isInitialized = false
    
    // 模型配置
    private var modelPath: String = ""
    private var numThreads: Int = 4
    private var useGpu: Boolean = false
    private var inputSize: Int = 256
    
    // 模型信息
    data class ModelInfo(
        val inputWidth: Int,
        val inputHeight: Int,
        val numParameters: Int
    )
    
    /**
     * 初始化引擎
     */
    fun initialize(
        context: Context,
        modelFileName: String = "digital_human.onnx",
        numThreads: Int = 4,
        useGpu: Boolean = false,
        inputSize: Int = 256
    ): Boolean {
        this.numThreads = numThreads
        this.useGpu = useGpu
        this.inputSize = inputSize
        
        // 复制模型文件到缓存目录
        modelPath = copyModelFromAssets(context, modelFileName)
        
        if (modelPath.isEmpty()) {
            Log.e(TAG, "Failed to copy model file")
            return false
        }
        
        // 初始化Native引擎
        isInitialized = nativeInitialize(modelPath, numThreads, useGpu, inputSize)
        
        if (isInitialized) {
            Log.i(TAG, "Engine initialized successfully")
        } else {
            Log.e(TAG, "Engine initialization failed")
        }
        
        return isInitialized
    }
    
    /**
     * 执行推理
     */
    fun infer(
        inputBitmap: Bitmap,
        audioFeatures: FloatArray
    ): Bitmap? {
        if (!isInitialized) {
            Log.e(TAG, "Engine not initialized")
            return null
        }
        
        // 确保输入Bitmap格式正确
        val input = inputBitmap.copy(Bitmap.Config.ARGB_8888, true)
        
        // 创建输出Bitmap
        val output = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888)
        
        // 执行Native推理
        val success = nativeInfer(input, audioFeatures, output)
        
        if (!success) {
            Log.e(TAG, "Inference failed")
            input.recycle()
            output.recycle()
            return null
        }
        
        input.recycle()
        return output
    }
    
    /**
     * 流式推理（单帧）
     */
    fun inferStream(
        referenceBitmap: Bitmap,
        audioFeature: FloatArray
    ): Bitmap? {
        return infer(referenceBitmap, audioFeature)
    }
    
    /**
     * 获取模型信息
     */
    fun getModelInfo(): ModelInfo? {
        if (!isInitialized) return null
        
        val info = IntArray(3)
        nativeGetModelInfo(info)
        
        return ModelInfo(
            inputWidth = info[0],
            inputHeight = info[1],
            numParameters = info[2]
        )
    }
    
    /**
     * 释放资源
     */
    fun release() {
        if (isInitialized) {
            nativeRelease()
            isInitialized = false
            Log.i(TAG, "Engine released")
        }
    }
    
    /**
     * 检查是否已初始化
     */
    fun isReady(): Boolean = isInitialized
    
    /**
     * 从Assets复制模型文件
     */
    private fun copyModelFromAssets(context: Context, fileName: String): String {
        val cacheDir = File(context.cacheDir, "models")
        if (!cacheDir.exists()) {
            cacheDir.mkdirs()
        }
        
        val destFile = File(cacheDir, fileName)
        
        // 如果文件已存在，直接返回路径
        if (destFile.exists()) {
            return destFile.absolutePath
        }
        
        try {
            context.assets.open(fileName).use { input ->
                FileOutputStream(destFile).use { output ->
                    input.copyTo(output)
                }
            }
            Log.i(TAG, "Model copied to: ${destFile.absolutePath}")
            return destFile.absolutePath
        } catch (e: Exception) {
            Log.e(TAG, "Failed to copy model: ${e.message}")
            return ""
        }
    }
    
    // Native方法声明
    private external fun nativeInitialize(
        modelPath: String,
        numThreads: Int,
        useGpu: Boolean,
        inputSize: Int
    ): Boolean
    
    private external fun nativeInfer(
        inputBitmap: Bitmap,
        audioFeatures: FloatArray,
        outputBitmap: Bitmap
    ): Boolean
    
    private external fun nativeRelease()
    
    private external fun nativeGetModelInfo(info: IntArray)
    
    // 加载Native库
    init {
        System.loadLibrary("digital_human_inference")
    }
}
