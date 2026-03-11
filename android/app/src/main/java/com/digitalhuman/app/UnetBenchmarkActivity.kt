package com.digitalhuman.app

import android.content.Context
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.io.FileOutputStream
import java.nio.ShortBuffer

class UnetBenchmarkActivity : AppCompatActivity() {

    companion object {
        private const val LOG_TAG = "UnetBenchmark"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val btn = findViewById<Button>(R.id.btnRunBenchmark)
        val txt = findViewById<TextView>(R.id.txtResult)

        btn.setOnClickListener {
            txt.text = "Running U-Net benchmark (FP16)..."
            Thread {
                try {
                    val ms = runUnetBenchmark(this, 50)
                    val fps = 1000f / ms
                    Log.i(
                        LOG_TAG,
                        "U-Net FP16 benchmark: avg=%.3f ms/frame (%.2f FPS)".format(ms, fps)
                    )
                    runOnUiThread {
                        txt.text = "U-Net FP16: %.2f ms/frame (%.1f FPS)".format(ms, fps)
                    }
                } catch (e: Exception) {
                    Log.e(LOG_TAG, "Benchmark failed", e)
                    runOnUiThread {
                        txt.text = "Benchmark failed: ${e.message}"
                    }
                }
            }.start()
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
     * 在当前设备上对 U-Net ONNX 进行前向推理基准测试。
     * 需要将 unet_wenet_160_fp16.onnx 放到 app/src/main/assets 下。
     */
    private fun runUnetBenchmark(context: Context, iters: Int): Float {
        val env = OrtEnvironment.getEnvironment()
        val modelFile = copyAssetToCache(context, "unet_wenet_160_fp16.onnx")
        val session: OrtSession = env.createSession(modelFile.absolutePath, OrtSession.SessionOptions())

        // 预设输入形状：image [1,6,160,160], audio [1,128,16,32]
        val imgShape = longArrayOf(1, 6, 160, 160)
        val audioShape = longArrayOf(1, 128, 16, 32)
        val imgSize = imgShape.reduce { acc, v -> acc * v }.toInt()
        val audioSize = audioShape.reduce { acc, v -> acc * v }.toInt()

        // FP16 模型需要 tensor(float16)，Java 端用 ShortBuffer 表示 half。
        val imgHalf = ShortArray(imgSize) { 0 }
        val audioHalf = ShortArray(audioSize) { 0 }

        val imgBuffer = ShortBuffer.wrap(imgHalf)
        val audioBuffer = ShortBuffer.wrap(audioHalf)

        val imgTensor = OnnxTensor.createTensor(env, imgBuffer, imgShape)
        val audioTensor = OnnxTensor.createTensor(env, audioBuffer, audioShape)

        val inputNames = session.inputNames.toList()

        // 预热
        repeat(5) {
            session.run(mapOf(
                inputNames[0] to imgTensor,
                inputNames[1] to audioTensor
            )).close()
        }

        val iterations = if (iters > 0) iters else 50
        val t0 = System.nanoTime()
        repeat(iterations) {
            session.run(mapOf(
                inputNames[0] to imgTensor,
                inputNames[1] to audioTensor
            )).close()
        }
        val t1 = System.nanoTime()

        imgTensor.close()
        audioTensor.close()
        session.close()
        env.close()

        val avgNs = (t1 - t0) / iterations.toDouble()
        return (avgNs / 1_000_000.0).toFloat()
    }
}


