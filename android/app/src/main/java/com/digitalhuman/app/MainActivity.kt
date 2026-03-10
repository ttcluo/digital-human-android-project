package com.digitalhuman.app

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.digitalhuman.app.engine.AudioProcessor
import com.digitalhuman.app.engine.DigitalHumanEngine
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "MainActivity"
        private const val PERMISSION_REQUEST_CODE = 100
    }

    // UI组件
    private lateinit var previewView: com.digitalhuman.app.databinding.ActivityMainBinding
    private lateinit var outputImageView: ImageView
    private lateinit var referenceImageView: ImageView
    private lateinit var btnStart: Button
    private lateinit var btnStop: Button
    private lateinit var statusTextView: TextView
    private lateinit var fpsTextView: TextView

    // 引擎
    private lateinit var engine: DigitalHumanEngine
    private lateinit var audioProcessor: AudioProcessor

    // 状态
    private var isRunning = false
    private var referenceBitmap: Bitmap? = null
    
    // 相机
    private lateinit var cameraExecutor: ExecutorService
    private var cameraProvider: ProcessCameraProvider? = null
    
    // 性能统计
    private var frameCount = 0
    private var startTime = 0L

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // 初始化绑定
        previewView = com.digitalhuman.app.databinding.ActivityMainBinding.inflate(layoutInflater)
        setContentView(previewView.root)
        
        // 初始化组件
        outputImageView = previewView.outputImage
        referenceImageView = previewView.referenceImage
        btnStart = previewView.btnStart
        btnStop = previewView.btnStop
        statusTextView = previewView.statusText
        fpsTextView = previewView.fpsText
        
        // 初始化引擎
        engine = DigitalHumanEngine.getInstance()
        audioProcessor = AudioProcessor()
        
        // 初始化相机
        cameraExecutor = Executors.newSingleThreadExecutor()
        
        // 初始化推理引擎
        initEngine()
        
        // 设置按钮点击事件
        setupButtons()
        
        // 检查权限
        checkPermissions()
        
        // 加载参考图像
        loadReferenceImage()
    }
    
    private fun initEngine() {
        CoroutineScope(Dispatchers.Main).launch {
            val success = engine.initialize(
                context = this@MainActivity,
                modelFileName = "digital_human.onnx",
                numThreads = 4,
                useGpu = false,
                inputSize = 256
            )
            
            if (success) {
                statusTextView.text = "引擎已就绪"
                btnStart.isEnabled = true
            } else {
                statusTextView.text = "引擎初始化失败"
                btnStart.isEnabled = false
            }
        }
    }
    
    private fun setupButtons() {
        btnStart.setOnClickListener {
            startDigitalHuman()
        }
        
        btnStop.setOnClickListener {
            stopDigitalHuman()
        }
        
        btnStart.isEnabled = false
    }
    
    private fun startDigitalHuman() {
        if (isRunning) return
        
        // 初始化音频处理器
        if (!audioProcessor.initialize()) {
            Toast.makeText(this, "音频初始化失败", Toast.LENGTH_SHORT).show()
            return
        }
        
        // 设置音频回调
        audioProcessor.onAudioDataCallback = { audioFeatures ->
            processAudioFrame(audioFeatures)
        }
        
        // 开始录音
        audioProcessor.startRecording()
        
        // 开始相机
        startCamera()
        
        isRunning = true
        btnStart.isEnabled = false
        btnStop.isEnabled = true
        statusTextView.text = "运行中..."
        
        // 重置统计
        frameCount = 0
        startTime = System.currentTimeMillis()
    }
    
    private fun stopDigitalHuman() {
        if (!isRunning) return
        
        // 停止音频
        audioProcessor.stopRecording()
        
        // 停止相机
        stopCamera()
        
        isRunning = false
        btnStart.isEnabled = true
        btnStop.isEnabled = false
        statusTextView.text = "已停止"
    }
    
    private fun processAudioFrame(audioFeatures: FloatArray) {
        referenceBitmap?.let { ref ->
            CoroutineScope(Dispatchers.Default).launch {
                val output = engine.inferStream(ref, audioFeatures)
                
                output?.let {
                    withContext(Dispatchers.Main) {
                        outputImageView.setImageBitmap(it)
                        
                        // 更新FPS
                        frameCount++
                        val elapsed = System.currentTimeMillis() - startTime
                        if (elapsed > 1000) {
                            val fps = frameCount * 1000f / elapsed
                            fpsTextView.text = "FPS: %.1f".format(fps)
                            frameCount = 0
                            startTime = System.currentTimeMillis()
                        }
                    }
                }
            }
        }
    }
    
    private fun loadReferenceImage() {
        // 加载默认参考图像
        referenceBitmap = Bitmap.createBitmap(256, 256, Bitmap.Config.ARGB_8888)
        referenceImageView.setImageBitmap(referenceBitmap)
    }
    
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        
        cameraProviderFuture.addListener({
            cameraProvider = cameraProviderFuture.get()
            
            // 预览
            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(previewView.cameraPreview.surfaceProvider)
                }
            
            // 图像分析
            val imageAnalysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { imageProxy ->
                        processImage(imageProxy)
                    }
                }
            
            // 绑定
            try {
                cameraProvider?.unbindAll()
                cameraProvider?.bindToLifecycle(
                    this,
                    CameraSelector.DEFAULT_FRONT_CAMERA,
                    preview,
                    imageAnalysis
                )
            } catch (e: Exception) {
                Log.e(TAG, "Camera binding failed", e)
            }
            
        }, ContextCompat.getMainExecutor(this))
    }
    
    private fun stopCamera() {
        cameraProvider?.unbindAll()
    }
    
    private fun processImage(imageProxy: ImageProxy) {
        // 获取图像
        val bitmap = imageProxy.toBitmap()
        
        // 调整大小
        val scaled = Bitmap.createScaledBitmap(bitmap, 256, 256, true)
        
        // 保存为参考图像
        referenceBitmap = scaled
        
        runOnUiThread {
            referenceImageView.setImageBitmap(scaled)
        }
        
        imageProxy.close()
    }
    
    private fun checkPermissions() {
        val permissions = arrayOf(
            Manifest.permission.CAMERA,
            Manifest.permission.RECORD_AUDIO
        )
        
        val needPermissions = permissions.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }
        
        if (needPermissions.isNotEmpty()) {
            ActivityCompat.requestPermissions(
                this,
                needPermissions.toTypedArray(),
                PERMISSION_REQUEST_CODE
            )
        }
    }
    
    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        
        if (requestCode == PERMISSION_REQUEST_CODE) {
            val allGranted = grantResults.all { it == PackageManager.PERMISSION_GRANTED }
            
            if (!allGranted) {
                Toast.makeText(this, "需要相机和音频权限", Toast.LENGTH_LONG).show()
            }
        }
    }
    
    override fun onDestroy() {
        super.onDestroy()
        
        // 释放资源
        engine.release()
        audioProcessor.release()
        cameraExecutor.shutdown()
    }
}
