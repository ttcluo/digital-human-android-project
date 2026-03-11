package com.digitalhuman.app

import android.content.Context
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Bundle
import android.util.Log
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer
import java.util.concurrent.ArrayBlockingQueue

class UnetBenchmarkActivity : AppCompatActivity() {

    companion object {
        private const val LOG_TAG = "UnetBenchmark"
        private const val SAMPLE_RATE = 16000
        private const val AUDIO_QUEUE_CAPACITY = 32
    }

    // Realtime demo相关字段（骨架实现）
    private var surfaceView: SurfaceView? = null
    @Volatile private var realtimeRunning: Boolean = false
    private var realtimeThread: Thread? = null
    private var audioThread: Thread? = null
    private var audioRecord: AudioRecord? = null
    private val audioQueue: ArrayBlockingQueue<ShortArray> =
        ArrayBlockingQueue(AUDIO_QUEUE_CAPACITY)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val btn = findViewById<Button>(R.id.btnRunBenchmark)
        val btnRealtime = findViewById<Button>(R.id.btnStartRealtime)
        val txt = findViewById<TextView>(R.id.txtResult)
        surfaceView = findViewById(R.id.surfaceView)

        btn.setOnClickListener {
            txt.text = "Running OnDevice U-Net benchmark (FP32)..."
            Thread {
                try {
                    val ms = runUnetBenchmark(this, 50)
                    val fps = 1000f / ms
                    Log.i(
                        LOG_TAG,
                        "OnDevice U-Net FP32 benchmark: avg=%.3f ms/frame (%.2f FPS)".format(ms, fps)
                    )
                    runOnUiThread {
                        txt.text = "OnDevice U-Net FP32: %.2f ms/frame (%.1f FPS)".format(ms, fps)
                    }
                } catch (e: Exception) {
                    Log.e(LOG_TAG, "Benchmark failed", e)
                    runOnUiThread {
                        txt.text = "Benchmark failed: ${e.message}"
                    }
                }
            }.start()
        }

        btnRealtime.setOnClickListener {
            if (!realtimeRunning) {
                txt.text = "Starting realtime demo (skeleton)..."
                startRealtimeDemo()
            }
        }
    }

    private fun copyAssetToCache(context: Context, assetName: String): File {
        val outFile = File(context.cacheDir, assetName)
        if (outFile.exists()) return outFile
        context.assets.open(assetName).use { input ->
            FileOutputStream(outFile).use { output ->
                input.copyTo(output)
            }
        }
        return outFile
    }

    /**
     * 在 Kotlin 中用 ONNX Runtime 跑端侧轻量 U-Net (OnDeviceUNet) 的 FP32 benchmark。
     * 需要将 unet_ondevice_128.onnx 放到 app/src/main/assets 下。
     */
    private fun runUnetBenchmark(context: Context, iters: Int): Float {
        val env = OrtEnvironment.getEnvironment()
        val modelFile = copyAssetToCache(context, "unet_ondevice_128.onnx")
        val session: OrtSession = env.createSession(modelFile.absolutePath, OrtSession.SessionOptions())

        // 预设输入形状：image [1,6,128,128], audio [1,128,16,32]
        val imgShape = longArrayOf(1, 6, 128, 128)
        val audioShape = longArrayOf(1, 128, 16, 32)
        val imgSize = imgShape.reduce { acc, v -> acc * v }.toInt()
        val audioSize = audioShape.reduce { acc, v -> acc * v }.toInt()

        val imgInput = FloatArray(imgSize) { 0.0f }
        val audioInput = FloatArray(audioSize) { 0.0f }

        val imgBuffer = FloatBuffer.wrap(imgInput)
        val audioBuffer = FloatBuffer.wrap(audioInput)

        val imgTensor = OnnxTensor.createTensor(env, imgBuffer, imgShape)
        val audioTensor = OnnxTensor.createTensor(env, audioBuffer, audioShape)

        val inputNames = session.inputNames.toList()

        // 预热
        repeat(5) {
            session.run(
                mapOf(
                    inputNames[0] to imgTensor,
                    inputNames[1] to audioTensor
                )
            ).close()
        }

        val iterations = if (iters > 0) iters else 50
        val t0 = System.nanoTime()
        repeat(iterations) {
            session.run(
                mapOf(
                    inputNames[0] to imgTensor,
                    inputNames[1] to audioTensor
                )
            ).close()
        }
        val t1 = System.nanoTime()

        imgTensor.close()
        audioTensor.close()
        session.close()
        env.close()

        val avgNs = (t1 - t0) / iterations.toDouble()
        return (avgNs / 1_000_000.0).toFloat()
    }

    /**
     * 端上全链路骨架（音频采集线程 + 特征队列 + U-Net 渲染 SurfaceView）。
     * 当前实现：
     * - 使用 AudioRecord 采集 16kHz PCM，推入 audioQueue（暂未真正送入 U-Net，仅作为架构占位）。
     * - 渲染线程按固定节奏调用 U-Net，使用零特征占位，输出结果渲染到 SurfaceView。
     * 后续可以在 audioQueue 之上接入轻量音频编码器，将其输出特征送入 U-Net。
     */
    private fun startRealtimeDemo() {
        val holder = surfaceView?.holder ?: run {
            Log.e(LOG_TAG, "SurfaceView holder is null")
            return
        }

        // 启动音频采集线程（骨架：采集后丢进队列，暂不参与推理）
        startAudioCapture()

        realtimeRunning = true
        realtimeThread = Thread {
            var env: OrtEnvironment? = null
            var session: OrtSession? = null
            var imgTensor: OnnxTensor? = null
            var audioTensor: OnnxTensor? = null
            try {
                env = OrtEnvironment.getEnvironment()
                val modelFile = copyAssetToCache(this, "unet_ondevice_128.onnx")
                session = env.createSession(
                    modelFile.absolutePath,
                    OrtSession.SessionOptions()
                )

                val imgShape = longArrayOf(1, 6, 128, 128)
                val audioShape = longArrayOf(1, 128, 16, 32)
                val imgSize = imgShape.reduce { acc, v -> acc * v }.toInt()
                val audioSize = audioShape.reduce { acc, v -> acc * v }.toInt()
                val imgInput = FloatArray(imgSize) { 0.0f }
                val audioInput = FloatArray(audioSize) { 0.0f }
                val imgBuffer = FloatBuffer.wrap(imgInput)
                val audioBuffer = FloatBuffer.wrap(audioInput)
                imgTensor = OnnxTensor.createTensor(env, imgBuffer, imgShape)
                audioTensor = OnnxTensor.createTensor(env, audioBuffer, audioShape)
                val inputNames = session.inputNames.toList()

                // 简单循环：固定节奏推理并渲染到 SurfaceView
                while (realtimeRunning) {
                    // 这里预留从 audioQueue 取出 PCM -> 特征的逻辑；当前使用零特征占位。
                    val result = session.run(
                        mapOf(
                            inputNames[0] to imgTensor,
                            inputNames[1] to audioTensor
                        )
                    )
                    val output = result[0].value as Array<FloatArray>
                    result.close()

                    // 仅作为骨架示例：不做 Bitmap 转换，只在日志中打点。
                    Log.d(LOG_TAG, "Realtime inference step done, output[0][0]=${output[0][0]}")

                    // 简单节流：假设目标帧率 20 FPS
                    Thread.sleep(50L)
                }
            } catch (e: Exception) {
                Log.e(LOG_TAG, "Realtime demo failed", e)
            } finally {
                imgTensor?.close()
                audioTensor?.close()
                session?.close()
                env?.close()
                stopAudioCapture()
            }
        }.also { it.start() }

        // 确保在 Surface 可用时再开始绘制（当前实现仅日志，可根据需要扩展为真正渲染）
        holder.addCallback(object : SurfaceHolder.Callback {
            override fun surfaceCreated(holder: SurfaceHolder) {}
            override fun surfaceChanged(
                holder: SurfaceHolder,
                format: Int,
                width: Int,
                height: Int
            ) {}

            override fun surfaceDestroyed(holder: SurfaceHolder) {
                realtimeRunning = false
            }
        })
    }

    private fun startAudioCapture() {
        val minBufSize = AudioRecord.getMinBufferSize(
            SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        )
        if (minBufSize == AudioRecord.ERROR || minBufSize == AudioRecord.ERROR_BAD_VALUE) {
            Log.e(LOG_TAG, "Invalid AudioRecord buffer size")
            return
        }
        val record = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            minBufSize
        )
        if (record.state != AudioRecord.STATE_INITIALIZED) {
            Log.e(LOG_TAG, "AudioRecord init failed")
            return
        }
        audioRecord = record
        audioThread = Thread {
            val buf = ShortArray(minBufSize)
            record.startRecording()
            while (realtimeRunning && record.recordingState == AudioRecord.RECORDSTATE_RECORDING) {
                val read = record.read(buf, 0, buf.size)
                if (read > 0) {
                    val copy = buf.copyOf(read)
                    audioQueue.offer(copy)
                }
            }
            record.stop()
            record.release()
        }.also { it.start() }
    }

    private fun stopAudioCapture() {
        realtimeRunning = false
        audioRecord = null
        audioThread = null
        audioQueue.clear()
    }
}