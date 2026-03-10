package com.digitalhuman.app.engine

import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Handler
import android.os.Looper
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs

/**
 * 音频处理器
 * 负责音频录制和特征提取
 */
class AudioProcessor(private val sampleRate: Int = 16000) {

    companion object {
        private const val TAG = "AudioProcessor"
        private const val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO
        private const val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT
    }

    private var audioRecord: AudioRecord? = null
    private var isRecording = false
    
    // 音频缓冲区
    private val bufferSize: Int = AudioRecord.getMinBufferSize(sampleRate, CHANNEL_CONFIG, AUDIO_FORMAT)
    
    // 回调
    var onAudioDataCallback: ((FloatArray) -> Unit)? = null
    
    // Handler for callbacks on main thread
    private val mainHandler = Handler(Looper.getMainLooper())
    
    /**
     * 初始化AudioRecord
     */
    fun initialize(): Boolean {
        return try {
            audioRecord = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                sampleRate,
                CHANNEL_CONFIG,
                AUDIO_FORMAT,
                bufferSize
            )
            
            if (audioRecord?.state != AudioRecord.STATE_INITIALIZED) {
                return false
            }
            
            true
        } catch (e: SecurityException) {
            false
        }
    }
    
    /**
     * 开始录音
     */
    fun startRecording(): Boolean {
        if (isRecording) return true
        
        return try {
            audioRecord?.startRecording()
            isRecording = true
            
            // 启动录音线程
            Thread { recordingLoop() }.start()
            
            true
        } catch (e: Exception) {
            false
        }
    }
    
    /**
     * 停止录音
     */
    fun stopRecording() {
        isRecording = false
        audioRecord?.stop()
    }
    
    /**
     * 释放资源
     */
    fun release() {
        stopRecording()
        audioRecord?.release()
        audioRecord = null
    }
    
    /**
     * 录音循环
     */
    private fun recordingLoop() {
        val buffer = ShortArray(bufferSize)
        
        while (isRecording) {
            val readSize = audioRecord?.read(buffer, 0, bufferSize) ?: 0
            
            if (readSize > 0) {
                // 转换为Float并归一化
                val floatData = ShortArray(readSize) { i -> 
                    buffer[i].toFloat() / Short.MAX_VALUE 
                }
                
                // 提取音频特征
                val features = extractFeatures(floatData)
                
                // 回调到主线程
                mainHandler.post {
                    onAudioDataCallback?.invoke(features)
                }
            }
        }
    }
    
    /**
     * 提取音频特征（简化版MFCC）
     * 实际使用时应替换为更复杂的特征提取
     */
    fun extractFeatures(audioData: ShortArray): FloatArray {
        val frameSize = 512
        val hopSize = 256
        val numFrames = (audioData.size - frameSize) / hopSize
        
        // 简化的特征提取：返回能量和基本统计
        val features = FloatArray(256)
        
        for (i in 0 until minOf(numFrames, 50)) {
            val start = i * hopSize
            var energy = 0f
            
            for (j in start until minOf(start + frameSize, audioData.size)) {
                energy += abs(audioData[j])
            }
            
            // 归一化能量
            val normalizedEnergy = energy / frameSize
            
            // 填充特征数组
            for (k in 0 until 5) {
                features[i * 5 + k] = normalizedEnergy
            }
        }
        
        return features
    }
    
    /**
     * 提取音频特征（从WAV文件）
     */
    suspend fun extractFeaturesFromFile(audioPath: String): FloatArray = withContext(Dispatchers.IO) {
        // 简化的实现
        // 实际应该使用librosa或类似的库
        FloatArray(256) { (Math.random() * 2 - 1).toFloat() }
    }
    
    /**
     * 从字节数组提取特征
     */
    fun extractFeaturesFromBytes(audioBytes: ByteArray): FloatArray {
        // 转换为Short数组
        val byteBuffer = ByteBuffer.wrap(audioBytes)
        byteBuffer.order(ByteOrder.LITTLE_ENDIAN)
        
        val shortData = ShortArray(audioBytes.size / 2)
        for (i in shortData.indices) {
            shortData[i] = byteBuffer.short
        }
        
        return extractFeatures(shortData)
    }
    
    /**
     * 平滑音频特征（用于流式推理）
     */
    fun smoothFeatures(current: FloatArray, previous: FloatArray, alpha: Float = 0.7f): FloatArray {
        return FloatArray(current.size) { i ->
            alpha * current[i] + (1 - alpha) * previous[i]
        }
    }
    
    /**
     * 生成静音帧特征
     */
    fun generateSilenceFeatures(): FloatArray {
        return FloatArray(256) { 0f }
    }
}
