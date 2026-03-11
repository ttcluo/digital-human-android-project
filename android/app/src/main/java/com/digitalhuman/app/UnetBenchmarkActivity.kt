package com.digitalhuman.app

import android.content.Context
import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.microsoft.onnxruntime.OnnxTensor
import com.microsoft.onnxruntime.OrtEnvironment
import com.microsoft.onnxruntime.OrtSession
import java.io.File
import java.io.FileOutputStream

class UnetBenchmarkActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val btn = findViewById<Button>(R.id.btnRunBenchmark)
        val txt = findViewById<TextView>(R.id.txtResult)

        btn.setOnClickListener {
            txt.text = "Running U-Net benchmark..."
            Thread {
                try {
                    val ms = runUnetBenchmark(this, 50)
                    val fps = 1000f / ms
                    runOnUiThread {
                        txt.text = "U-Net: %.2f ms/frame (%.1f FPS)".format(ms, fps)
                    }
                } catch (e: Exception) {
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
     * 需要将 unet_wenet_160.onnx 放到 app/src/main/assets 下。
     */
    private fun runUnetBenchmark(context: Context, iters: Int): Float {
        val env = OrtEnvironment.getEnvironment()
        val modelFile = copyAssetToCache(context, "unet_wenet_160.onnx")
        val session: OrtSession = env.createSession(modelFile.absolutePath, OrtSession.SessionOptions())

        // 预设输入形状：image [1,6,160,160], audio [1,128,16,32]
        val imgShape = longArrayOf(1, 6, 160, 160)
        val audioShape = longArrayOf(1, 128, 16, 32)
        val imgSize = imgShape.reduce { acc, v -> acc * v }.toInt()
        val audioSize = audioShape.reduce { acc, v -> acc * v }.toInt()

        val imgInput = FloatArray(imgSize) { 0.0f }
        val audioInput = FloatArray(audioSize) { 0.0f }

        val imgTensor = OnnxTensor.createTensor(env, imgInput, imgShape)
        val audioTensor = OnnxTensor.createTensor(env, audioInput, audioShape)

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


