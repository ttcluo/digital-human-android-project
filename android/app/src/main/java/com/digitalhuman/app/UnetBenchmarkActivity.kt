package com.digitalhuman.app

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.Rect
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
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.concurrent.ArrayBlockingQueue

class UnetBenchmarkActivity : AppCompatActivity() {

    companion object {
        private const val LOG_TAG = "UnetBenchmark"
        private const val SAMPLE_RATE = 16000
        private const val AUDIO_QUEUE_CAPACITY = 32
        private const val REQ_RECORD_AUDIO = 1001
    }

    // Realtime demo相关字段（骨架实现）
    private var surfaceView: SurfaceView? = null
    @Volatile private var realtimeRunning: Boolean = false
    private var realtimeThread: Thread? = null
    private var audioThread: Thread? = null
    private var audioRecord: AudioRecord? = null
    private val audioQueue: ArrayBlockingQueue<ShortArray> =
        ArrayBlockingQueue(AUDIO_QUEUE_CAPACITY)
    private var wenetFeatFrames: FloatArray? = null
    private var wenetFrameCount: Int = 0
    private var wenetFeatSize: Int = 0
    private var refFaceBitmap: Bitmap? = null

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
                if (!ensureRecordAudioPermission()) {
                    txt.text = "需要录音权限才能启动实时 demo"
                } else {
                    txt.text = "Starting realtime demo (skeleton)..."
                    startRealtimeDemo()
                }
            }
        }
    }

    private fun ensureRecordAudioPermission(): Boolean {
        val granted = ContextCompat.checkSelfPermission(
            this,
            Manifest.permission.RECORD_AUDIO
        ) == PackageManager.PERMISSION_GRANTED
        if (!granted) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.RECORD_AUDIO),
                REQ_RECORD_AUDIO
            )
        }
        return granted
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQ_RECORD_AUDIO) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                Log.i(LOG_TAG, "RECORD_AUDIO permission granted")
            } else {
                Log.e(LOG_TAG, "RECORD_AUDIO permission denied by user")
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

        // 预加载伪流式 WeNet 特征（如果存在的话）
        if (wenetFeatFrames == null) {
            try {
                loadWenetFeatStream(this, "wenet_feat_stream.bin")
            } catch (e: Exception) {
                Log.e(LOG_TAG, "loadWenetFeatStream failed, fallback to zero audio", e)
            }
        }

        realtimeRunning = true
        realtimeThread = Thread {
            val outH = 128
            val outW = 128
            var env: OrtEnvironment? = null
            var session: OrtSession? = null
            var imgTensor: OnnxTensor? = null
            var audioTensor: OnnxTensor? = null
            var bitmap: Bitmap? = null
            var pixels: IntArray? = null
            try {
                env = OrtEnvironment.getEnvironment()
                val modelFile = copyAssetToCache(this, "unet_ondevice_128.onnx")
                session = env.createSession(
                    modelFile.absolutePath,
                    OrtSession.SessionOptions()
                )

                val imgShape = longArrayOf(1, 6, outH.toLong(), outW.toLong())
                val audioShape = longArrayOf(1, 128, 16, 32)
                val imgSize = imgShape.reduce { acc, v -> acc * v }.toInt()
                val audioSize = audioShape.reduce { acc, v -> acc * v }.toInt()
                val imgInput = FloatArray(imgSize) { 0.0f }
                // 单帧音频特征缓冲区，形状 [1,128,16,32]
                val audioInput = FloatArray(audioSize) { 0.0f }

                // 尝试从 assets 加载一张 128x128 参考人脸图片，填入 6 通道图像输入（前 3 通道 + 后 3 通道复用同一张图）。
                // 需要你在 app/src/main/assets 下自行放置一张 ref_face_128.png。
                try {
                    loadRefImageToTensor(this, "ref_face_128.png", imgInput, outW, outH)
                } catch (e: Exception) {
                    Log.e(LOG_TAG, "loadRefImageToTensor failed, fallback to zeros", e)
                }

                val imgBuffer = FloatBuffer.wrap(imgInput)
                val audioBuffer = FloatBuffer.wrap(audioInput)
                imgTensor = OnnxTensor.createTensor(env, imgBuffer, imgShape)
                audioTensor = OnnxTensor.createTensor(env, audioBuffer, audioShape)
                val inputNames = session.inputNames.toList()

                bitmap = Bitmap.createBitmap(outW, outH, Bitmap.Config.ARGB_8888)
                pixels = IntArray(outW * outH)
                Log.i(LOG_TAG, "Realtime demo init: outH=$outH, outW=$outW, imgSize=$imgSize, audioSize=$audioSize")

                var audioFrameIndex = 0
                var frameCounter = 0

                while (realtimeRunning) {
                    // 每帧从预生成的 WeNet 特征序列中取出一帧填充 audioInput；如果没有则保持全 0。
                    val featFrames = wenetFeatFrames
                    if (featFrames != null && wenetFrameCount > 0 && wenetFeatSize > 0) {
                        val frame = audioFrameIndex % wenetFrameCount
                        val base = frame * wenetFeatSize
                        var i = 0
                        while (i < wenetFeatSize && i < audioInput.size) {
                            audioInput[i] = featFrames[base + i]
                            i++
                        }
                        audioFrameIndex++
                    }

                    val result = session.run(
                        mapOf(
                            inputNames[0] to imgTensor,
                            inputNames[1] to audioTensor
                        )
                    )
                    @Suppress("UNCHECKED_CAST")
                    val output = result[0].value as Array<Array<Array<FloatArray>>>
                    result.close()

                    val bmp = bitmap
                    val px = pixels
                    if (bmp == null || px == null) {
                        Log.w(LOG_TAG, "Bitmap or pixels is null, skip rendering")
                    } else if (output.isEmpty()
                        || output[0].isEmpty()
                        || output[0][0].isEmpty()
                        || output[0][0][0].isEmpty()
                    ) {
                        Log.w(LOG_TAG, "Output tensor is empty, skip rendering")
                    } else {
                        // 每隔若干帧打印一次输出范围，确认是否全部为 0 或接近 0
                        if (frameCounter % 10 == 0) {
                            var minVal = Float.POSITIVE_INFINITY
                            var maxVal = Float.NEGATIVE_INFINITY
                            val c0dbg = output[0][0]
                            var yDbg = 0
                            while (yDbg < outH) {
                                var xDbg = 0
                                while (xDbg < outW) {
                                    val v = c0dbg[yDbg][xDbg]
                                    if (v < minVal) minVal = v
                                    if (v > maxVal) maxVal = v
                                    xDbg++
                                }
                                yDbg++
                            }
                            Log.d(LOG_TAG, "U-Net output[0] range: min=$minVal, max=$maxVal")
                        }

                        val c0 = output[0][0]
                        val c1 = output[0][1]
                        val c2 = output[0][2]

                        val base = refFaceBitmap
                        var idx = 0
                        var y = 0
                        while (y < outH) {
                            var x = 0
                            while (x < outW) {
                                val fr = c0[y][x].coerceIn(0.0f, 1.0f)
                                val fg = c1[y][x].coerceIn(0.0f, 1.0f)
                                val fb = c2[y][x].coerceIn(0.0f, 1.0f)

                                var outR: Int
                                var outG: Int
                                var outB: Int

                                if (base != null) {
                                    val bc = base.getPixel(x, y)
                                    val br = Color.red(bc) / 255.0f
                                    val bg = Color.green(bc) / 255.0f
                                    val bb = Color.blue(bc) / 255.0f
                                    // 简单线性混合：大部分来自 U-Net 预测，小部分保留底图细节
                                    outR = ((fr * 0.7f + br * 0.3f) * 255.0f).toInt()
                                    outG = ((fg * 0.7f + bg * 0.3f) * 255.0f).toInt()
                                    outB = ((fb * 0.7f + bb * 0.3f) * 255.0f).toInt()
                                } else {
                                    outR = (fr * 255.0f).toInt()
                                    outG = (fg * 255.0f).toInt()
                                    outB = (fb * 255.0f).toInt()
                                }

                                px[idx++] =
                                    (0xFF shl 24) or
                                        (outR.coerceIn(0, 255) shl 16) or
                                        (outG.coerceIn(0, 255) shl 8) or
                                        outB.coerceIn(0, 255)
                                x++
                            }
                            y++
                        }
                        bmp.setPixels(px, 0, outW, 0, 0, outW, outH)

                        val canvas = holder.lockCanvas()
                        if (canvas != null) {
                            try {
                                canvas.drawColor(Color.BLACK)
                                val dest = Rect(0, 0, canvas.width, canvas.height)
                                canvas.drawBitmap(bmp, null, dest, null)
                            } finally {
                                holder.unlockCanvasAndPost(canvas)
                            }
                        }
                    }

                    Log.d(LOG_TAG, "Realtime inference step done")
                    frameCounter++
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

    /**
     * 从 assets 中加载通过 export_wenet_feat_for_android.py 生成的伪流式 WeNet 特征：
     * 文件格式：
     * - 前 4 个 int32: [T, C, H, W]，其中 C=128, H=16, W=32
     * - 后面为 T*C*H*W 个 float32，按 C-H-W 展开。
     */
    private fun loadWenetFeatStream(context: Context, assetName: String) {
        context.assets.open(assetName).use { input ->
            val headerBytes = ByteArray(4 * 4)
            var read = input.read(headerBytes)
            if (read != headerBytes.size) {
                throw IllegalStateException("读取 WeNet 特征头部失败，read=$read")
            }
            val headerBuf = ByteBuffer.wrap(headerBytes).order(ByteOrder.LITTLE_ENDIAN)
            val T = headerBuf.int
            val C = headerBuf.int
            val H = headerBuf.int
            val W = headerBuf.int
            if (C != 128 || H != 16 || W != 32) {
                throw IllegalStateException("WeNet 特征维度不匹配: C=$C,H=$H,W=$W, 期望 128,16,32")
            }
            val featSize = C * H * W
            // 为避免在手机上一次性加载过长语音导致 OOM，这里只截取前 N 帧。
            val framesToLoad = minOf(T, 200) // 例如最多加载约 200 帧
            val total = framesToLoad * featSize
            val dataBytes = ByteArray(total * 4)
            var offset = 0
            while (offset < dataBytes.size) {
                val r = input.read(dataBytes, offset, dataBytes.size - offset)
                if (r <= 0) break
                offset += r
            }
            if (offset != dataBytes.size) {
                throw IllegalStateException("读取 WeNet 特征体失败: 期望 ${dataBytes.size} 字节, 实际 $offset")
            }
            val floatBuf =
                ByteBuffer.wrap(dataBytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
            val arr = FloatArray(total)
            floatBuf.get(arr)
            wenetFeatFrames = arr
            wenetFrameCount = framesToLoad
            wenetFeatSize = featSize
            Log.i(
                LOG_TAG,
                "Loaded WeNet feat stream: T=$T, useFrames=$framesToLoad, C=$C, H=$H, W=$W, total=${arr.size}"
            )
        }
    }

    /**
     * 将 assets 中的一张 128x128 人脸图片编码进 U-Net 的 6 通道图像输入：
     * - Bitmap 读取后缩放到 (width,height)
     * - 归一化到 [0,1]
     * - 同一张图复制到前 3 通道和后 3 通道（模拟两帧图像）。
     */
    private fun loadRefImageToTensor(
        context: Context,
        assetName: String,
        imgInput: FloatArray,
        width: Int,
        height: Int
    ) {
        context.assets.open(assetName).use { input ->
            val raw = BitmapFactory.decodeStream(input)
            val bmp = Bitmap.createScaledBitmap(raw, width, height, true)
            raw.recycle()

            // 保留一份用于后续将 U-Net patch 贴回整张人脸
            refFaceBitmap = bmp

            val hw = height * width

            // 构造两张图：一张原始参考图，一张中间区域涂黑的 masked 图，用于模拟训练时的 [参考图, 被遮挡图] 输入。
            val masked = bmp.copy(Bitmap.Config.ARGB_8888, true)
            val maskCanvas = android.graphics.Canvas(masked)
            val margin = (width * 0.08f).toInt()   // 近似 (5/160) 的相对边距
            val rect = android.graphics.Rect(
                margin,
                margin,
                width - margin,
                height - margin
            )
            maskCanvas.drawRect(rect, android.graphics.Paint().apply { color = Color.BLACK })

            var y = 0
            while (y < height) {
                var x = 0
                while (x < width) {
                    val cRef = bmp.getPixel(x, y)
                    val cMask = masked.getPixel(x, y)

                    val rRef = (Color.red(cRef) / 255.0f).coerceIn(0.0f, 1.0f)
                    val gRef = (Color.green(cRef) / 255.0f).coerceIn(0.0f, 1.0f)
                    val bRef = (Color.blue(cRef) / 255.0f).coerceIn(0.0f, 1.0f)

                    val rMask = (Color.red(cMask) / 255.0f).coerceIn(0.0f, 1.0f)
                    val gMask = (Color.green(cMask) / 255.0f).coerceIn(0.0f, 1.0f)
                    val bMask = (Color.blue(cMask) / 255.0f).coerceIn(0.0f, 1.0f)

                    val idx = y * width + x

                    // 通道顺序：[参考 B,G,R, 被遮挡 B,G,R]，与训练时 BGR*2 对齐。
                    imgInput[0 * hw + idx] = bRef
                    imgInput[1 * hw + idx] = gRef
                    imgInput[2 * hw + idx] = rRef
                    imgInput[3 * hw + idx] = bMask
                    imgInput[4 * hw + idx] = gMask
                    imgInput[5 * hw + idx] = rMask

                    x++
                }
                y++
            }
        }
    }

    private fun startAudioCapture() {
        // 录音相关逻辑暂时关闭，避免对当前端上全链路调试产生干扰。
        Log.i(LOG_TAG, "startAudioCapture() skipped (audio disabled)")
        return
        /*
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
        */
    }

    private fun stopAudioCapture() {
        realtimeRunning = false
        audioRecord = null
        audioThread = null
        audioQueue.clear()
    }
}