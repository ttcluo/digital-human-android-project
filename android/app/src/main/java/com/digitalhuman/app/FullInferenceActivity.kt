package com.digitalhuman.app

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
import android.media.MediaMuxer
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.util.Log
import android.widget.Button
import android.widget.ProgressBar
import android.widget.RadioGroup
import android.widget.TextView
import android.widget.Toast
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
import java.util.concurrent.atomic.AtomicBoolean

/**
 * 完整推理 Activity：移植服务端 inference.py 流程。
 * 输入：assets/dataset + assets/wenet_feat_stream.bin
 * 输出：完整视频 mp4（含口型驱动的人脸贴回全图）
 */
class FullInferenceActivity : AppCompatActivity() {

    companion object {
        private const val LOG_TAG = "FullInference"
        private const val REQ_STORAGE = 1001
        private const val WENET_FPS = 20
        private const val UNET_INPUT_SIZE = 128
        private const val CROP_168 = 168
        private const val PATCH_160 = 160
        private const val MASK_MARGIN = 5
        private const val MASK_RIGHT = 150
        private const val MASK_BOTTOM = 145
    }

    private lateinit var btnSelectAudio: Button
    private lateinit var btnRunInference: Button
    private lateinit var txtAudioPath: TextView
    private lateinit var txtProgress: TextView
    private lateinit var txtOutputPath: TextView
    private lateinit var progressBar: ProgressBar

    private var selectedAudioPath: String? = null
    private val running = AtomicBoolean(false)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_full_inference)

        btnSelectAudio = findViewById(R.id.btnSelectAudio)
        btnRunInference = findViewById(R.id.btnRunInference)
        txtAudioPath = findViewById(R.id.txtAudioPath)
        txtProgress = findViewById(R.id.txtProgress)
        txtOutputPath = findViewById(R.id.txtOutputPath)
        progressBar = findViewById(R.id.progressBar)

        initDefaultAudio()

        btnSelectAudio.setOnClickListener {
            if (ensureStoragePermission()) {
                val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
                    addCategory(Intent.CATEGORY_OPENABLE)
                    type = "audio/*"
                }
                try {
                    startActivityForResult(Intent.createChooser(intent, "选择音频"), REQ_STORAGE)
                } catch (e: Exception) {
                    Toast.makeText(this, "无法打开文件选择器: ${e.message}", Toast.LENGTH_SHORT).show()
                }
            } else {
                txtAudioPath.text = "默认: preview.mp3（需存储权限才能选择其他文件）"
            }
        }

        btnRunInference.setOnClickListener {
            if (running.get()) return@setOnClickListener
            val useUltralight160 = findViewById<RadioGroup>(R.id.radioModel).checkedRadioButtonId == R.id.radioUltralight
            runFullInference(useUltralight160)
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == REQ_STORAGE && resultCode == RESULT_OK && data?.data != null) {
            val uri = data.data!!
            selectedAudioPath = copyUriToCache(uri, "selected_audio.mp3")?.absolutePath
            txtAudioPath.text = selectedAudioPath ?: "已选择（见上方）"
        }
    }

    private fun initDefaultAudio() {
        try {
            val outFile = File(cacheDir, "preview.mp3")
            if (!outFile.exists()) {
                assets.open("preview.mp3").use { input ->
                    FileOutputStream(outFile).use { output -> input.copyTo(output) }
                }
            }
            selectedAudioPath = outFile.absolutePath
            txtAudioPath.text = "默认: preview.mp3"
        } catch (e: Exception) {
            Log.w(LOG_TAG, "preview.mp3 not in assets, using null", e)
            txtAudioPath.text = "未找到 preview.mp3，推理将输出无声视频"
        }
    }

    private fun copyUriToCache(uri: android.net.Uri, fileName: String): File? {
        return try {
            val outFile = File(cacheDir, fileName)
            contentResolver.openInputStream(uri)?.use { input ->
                FileOutputStream(outFile).use { output -> input.copyTo(output) }
            }
            outFile
        } catch (e: Exception) {
            Log.e(LOG_TAG, "copyUriToCache failed", e)
            null
        }
    }

    private fun ensureStoragePermission(): Boolean {
        val perm = if (Build.VERSION.SDK_INT >= 33) {
            Manifest.permission.READ_MEDIA_AUDIO
        } else {
            Manifest.permission.READ_EXTERNAL_STORAGE
        }
        val granted = ContextCompat.checkSelfPermission(this, perm) == PackageManager.PERMISSION_GRANTED
        if (!granted) {
            ActivityCompat.requestPermissions(this, arrayOf(perm), REQ_STORAGE)
        }
        return granted
    }

    private fun runFullInference(useUltralight160: Boolean = false) {
        running.set(true)
        btnRunInference.isEnabled = false
        progressBar.visibility = ProgressBar.VISIBLE
        progressBar.max = 100
        progressBar.progress = 0
        txtProgress.text = "准备中..."
        txtOutputPath.text = ""

        val suffix = if (useUltralight160) "ultralight" else "ondevice"
        Thread {
            try {
                val outFile = File(getOutputDir(), "full_inference_${suffix}_${System.currentTimeMillis()}.mp4")
                runInferencePipeline(outFile, useUltralight160)
                runOnUiThread {
                    txtProgress.text = "完成"
                    txtOutputPath.text = outFile.absolutePath
                    progressBar.progress = 100
                    Toast.makeText(this, "视频已保存: ${outFile.name}", Toast.LENGTH_LONG).show()
                }
            } catch (e: Exception) {
                Log.e(LOG_TAG, "Full inference failed", e)
                runOnUiThread {
                    txtProgress.text = "失败: ${e.message}"
                    Toast.makeText(this, "推理失败: ${e.message}", Toast.LENGTH_LONG).show()
                }
            } finally {
                running.set(false)
                runOnUiThread {
                    btnRunInference.isEnabled = true
                    progressBar.visibility = ProgressBar.GONE
                }
            }
        }.start()
    }

    private fun getOutputDir(): File {
        val dir = if (Build.VERSION.SDK_INT >= 29) {
            getExternalFilesDir(Environment.DIRECTORY_MOVIES) ?: filesDir
        } else {
            File(Environment.getExternalStorageDirectory(), "Movies/DigitalHuman").apply { mkdirs() }
        }
        return dir
    }

    private fun runInferencePipeline(outFile: File, useUltralight160: Boolean = false) {
        val datasetDir = File(filesDir, "dataset").apply { mkdirs() }
        copyAssetsDataset(datasetDir)

        val wenetFeat = loadWenetFeatStream(this, "wenet_feat_stream.bin")
        val (featFrames, featFrameCount, featSize) = wenetFeat
            ?: throw IllegalStateException("未找到 wenet_feat_stream.bin，请先运行 export_preview_for_android.sh")

        val imgDir = File(datasetDir, "full_body_img")
        val lmsDir = File(datasetDir, "landmarks")
        val imgFiles = imgDir.listFiles { _, name -> name.endsWith(".jpg") }?.sortedBy { it.nameWithoutExtension.toIntOrNull() ?: 0 }
            ?: throw IllegalStateException("dataset/full_body_img 为空，请运行 export_dataset_for_android.sh")
        val lenImg = imgFiles.size

        if (lenImg == 0) throw IllegalStateException("dataset 无图片")

        val exmPath = imgFiles[0].absolutePath
        val exmBmp = decodeImageFile(exmPath)
            ?: throw IllegalStateException("无法读取示例图: $exmPath")
        val outW = exmBmp.width
        val outH = exmBmp.height
        exmBmp.recycle()

        val env = OrtEnvironment.getEnvironment()
        val (modelAsset, inputSize) = if (useUltralight160) {
            "unet_wenet_160.onnx" to 160
        } else {
            "unet_ondevice_128.onnx" to 128
        }
        val modelFile = copyAssetToCache(this, modelAsset)
        val session = env.createSession(modelFile.absolutePath, OrtSession.SessionOptions())
        val inputNames = session.inputNames.toList()

        val imgShape = longArrayOf(1, 6, inputSize.toLong(), inputSize.toLong())
        val audioShape = longArrayOf(1, 128, 16, 32)
        val audioInput = FloatArray(audioShape.reduce { a, v -> a * v }.toInt())

        val totalFrames = featFrameCount
        var stepStride = 1
        var imgIdx = 0

        val videoEncoder = VideoEncoder(outW, outH, WENET_FPS)
        videoEncoder.start(outFile.absolutePath)

        for (i in 0 until totalFrames) {
            if (imgIdx > lenImg - 1) stepStride = -1
            if (imgIdx < 1) stepStride = 1
            imgIdx += stepStride

            val frameIdx = imgIdx.coerceIn(0, lenImg - 1)
            val imgPath = imgFiles[frameIdx].absolutePath
            val lmsPath = File(lmsDir, "${imgFiles[frameIdx].nameWithoutExtension}.lms").absolutePath

            val fullImg = decodeImageFile(imgPath)
                ?: throw IllegalStateException("无法读取: $imgPath")
            val lms = parseLandmarks(lmsPath)
            var (xmin, ymin, xmax, ymax) = getCropRect(lms)
            xmin = xmin.coerceIn(0, fullImg.width - 1)
            ymin = ymin.coerceIn(0, fullImg.height - 1)
            xmax = xmax.coerceIn(0, fullImg.width)
            ymax = ymax.coerceIn(0, fullImg.height)

            val cropW = (xmax - xmin).coerceAtLeast(1)
            val cropH = (ymax - ymin).coerceAtLeast(1)
            val cropBmp = Bitmap.createBitmap(fullImg, xmin, ymin, cropW, cropH)
            val crop168 = Bitmap.createScaledBitmap(cropBmp, CROP_168, CROP_168, true)
            cropBmp.recycle()

            val patch160 = Bitmap.createBitmap(crop168, 4, 4, PATCH_160, PATCH_160)
            val masked160 = patch160.copy(Bitmap.Config.ARGB_8888, true)
            Canvas(masked160).drawRect(
                MASK_MARGIN.toFloat(), MASK_MARGIN.toFloat(),
                MASK_RIGHT.toFloat(), MASK_BOTTOM.toFloat(),
                Paint().apply { color = Color.BLACK }
            )

            val imgInput = FloatArray(6 * inputSize * inputSize)
            fillSixChannelInput(patch160, masked160, imgInput, inputSize)
            patch160.recycle()
            masked160.recycle()

            val base = i * featSize
            for (j in audioInput.indices) {
                audioInput[j] = if (base + j < featFrames.size) featFrames[base + j] else 0f
            }

            val imgTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(imgInput), imgShape)
            val audioTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(audioInput), audioShape)

            val result = session.run(
                mapOf(inputNames[0] to imgTensor, inputNames[1] to audioTensor)
            )
            @Suppress("UNCHECKED_CAST")
            val output = result[0].value as Array<Array<Array<FloatArray>>>
            result.close()
            imgTensor.close()
            audioTensor.close()

            val pred160 = outputToBitmap(output, inputSize)
            val crop168Out = crop168.copy(Bitmap.Config.ARGB_8888, true)
            val canvas = Canvas(crop168Out)
            val noFilterPaint = Paint().apply { isFilterBitmap = false }
            canvas.drawBitmap(pred160, null, Rect(4, 4, 164, 164), noFilterPaint)
            pred160.recycle()

            val cropResized = Bitmap.createScaledBitmap(crop168Out, cropW, cropH, true)
            crop168Out.recycle()
            crop168.recycle()

            val outBmp = fullImg.copy(Bitmap.Config.ARGB_8888, true)
            Canvas(outBmp).drawBitmap(cropResized, xmin.toFloat(), ymin.toFloat(), null)
            cropResized.recycle()
            fullImg.recycle()

            if (i == 0) {
                try {
                    FileOutputStream(File(filesDir, "full_inference_frame0_debug.png")).use { fos ->
                        outBmp.compress(Bitmap.CompressFormat.PNG, 100, fos)
                    }
                    Log.i(LOG_TAG, "Debug frame saved: ${filesDir}/full_inference_frame0_debug.png")
                } catch (e: Exception) { Log.e(LOG_TAG, "Save debug frame failed", e) }
            }
            videoEncoder.encodeFrame(outBmp)
            outBmp.recycle()

            if (i % 10 == 0) {
                runOnUiThread {
                    progressBar.progress = (i * 100 / totalFrames).coerceAtMost(99)
                    txtProgress.text = "帧 $i / $totalFrames"
                }
            }
        }

        videoEncoder.stop()
        session.close()
        env.close()
    }

    private fun copyAssetsDataset(destDir: File) {
        val fullBodyDest = File(destDir, "full_body_img").apply { mkdirs() }
        val landmarksDest = File(destDir, "landmarks").apply { mkdirs() }
        try {
            assets.list("dataset/full_body_img")?.forEach { name ->
                assets.open("dataset/full_body_img/$name").use { input ->
                    FileOutputStream(File(fullBodyDest, name)).use { output -> input.copyTo(output) }
                }
            }
            assets.list("dataset/landmarks")?.forEach { name ->
                assets.open("dataset/landmarks/$name").use { input ->
                    FileOutputStream(File(landmarksDest, name)).use { output -> input.copyTo(output) }
                }
            }
        } catch (e: Exception) {
            Log.w(LOG_TAG, "copyAssetsDataset: ${e.message}")
        }
    }

    private fun decodeImageFile(path: String): Bitmap? {
        val opts = BitmapFactory.Options().apply {
            inPreferredConfig = Bitmap.Config.ARGB_8888
            inScaled = false
            inDither = false
        }
        return BitmapFactory.decodeFile(path, opts)
    }

    private fun copyAssetToCache(context: Context, assetName: String): File {
        val outFile = File(context.cacheDir, assetName)
        if (outFile.exists()) outFile.delete()
        context.assets.open(assetName).use { input ->
            FileOutputStream(outFile).use { output -> input.copyTo(output) }
        }
        return outFile
    }

    private fun loadWenetFeatStream(context: Context, assetName: String): Triple<FloatArray, Int, Int>? {
        return try {
            context.assets.open(assetName).use { input ->
                val header = ByteArray(16)
                if (input.read(header) != 16) return null
                val buf = ByteBuffer.wrap(header).order(ByteOrder.LITTLE_ENDIAN)
                val T = buf.int
                val C = buf.int
                val H = buf.int
                val W = buf.int
                if (C != 128 || H != 16 || W != 32) return null
                val featSize = C * H * W
                val total = T * featSize
                val data = ByteArray(total * 4)
                var offset = 0
                while (offset < data.size) {
                    val r = input.read(data, offset, data.size - offset)
                    if (r <= 0) break
                    offset += r
                }
                if (offset != data.size) return null
                val arr = FloatArray(total)
                ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(arr)
                Triple(arr, T, featSize)
            }
        } catch (e: Exception) {
            Log.e(LOG_TAG, "loadWenetFeatStream failed", e)
            null
        }
    }

    private fun parseLandmarks(path: String): Array<FloatArray> {
        val lines = File(path).readLines()
        return lines.map { line ->
            line.split(" ").map { it.toFloat() }.toFloatArray()
        }.toTypedArray()
    }

    private fun getCropRect(lms: Array<FloatArray>): IntArray {
        val xmin = lms[1][0].toInt()
        val ymin = lms[52][1].toInt()
        val xmax = lms[31][0].toInt()
        val width = xmax - xmin
        val ymax = ymin + width
        return intArrayOf(xmin, ymin, xmax, ymax)
    }

    private fun fillSixChannelInput(real: Bitmap, masked: Bitmap, out: FloatArray, inputSize: Int) {
        val realScaled = Bitmap.createScaledBitmap(real, inputSize, inputSize, true)
        val maskedScaled = Bitmap.createScaledBitmap(masked, inputSize, inputSize, true)
        val hw = inputSize * inputSize
        for (y in 0 until inputSize) {
            for (x in 0 until inputSize) {
                val idx = y * inputSize + x
                val cReal = realScaled.getPixel(x, y)
                val cMask = maskedScaled.getPixel(x, y)
                out[0 * hw + idx] = Color.blue(cReal) / 255f
                out[1 * hw + idx] = Color.green(cReal) / 255f
                out[2 * hw + idx] = Color.red(cReal) / 255f
                out[3 * hw + idx] = Color.blue(cMask) / 255f
                out[4 * hw + idx] = Color.green(cMask) / 255f
                out[5 * hw + idx] = Color.red(cMask) / 255f
            }
        }
        realScaled.recycle()
        maskedScaled.recycle()
    }

    /**
     * 将 pred 与原始图在边缘融合，消除方形马赛克边界感。
     * margin 像素内线性混合：中心 100% pred，边缘 100% 原图。
     */
    private fun blendPredWithOriginal(
        crop168: Bitmap,
        pred160: Bitmap,
        ox: Int,
        oy: Int,
        w: Int,
        h: Int
    ): Bitmap {
        val margin = 12
        val orig160 = Bitmap.createBitmap(crop168, ox, oy, w, h)
        val predPx = IntArray(w * h)
        val origPx = IntArray(w * h)
        pred160.getPixels(predPx, 0, w, 0, 0, w, h)
        orig160.getPixels(origPx, 0, w, 0, 0, w, h)
        orig160.recycle()
        val outPx = IntArray(w * h)
        for (y in 0 until h) {
            for (x in 0 until w) {
                val d = minOf(x, y, w - 1 - x, h - 1 - y)
                val alpha = if (d >= margin) 255 else (d * 255 / margin).coerceAtLeast(0)
                val p = predPx[y * w + x]
                val o = origPx[y * w + x]
                val r = (Color.red(p) * alpha + Color.red(o) * (255 - alpha)) / 255
                val g = (Color.green(p) * alpha + Color.green(o) * (255 - alpha)) / 255
                val b = (Color.blue(p) * alpha + Color.blue(o) * (255 - alpha)) / 255
                outPx[y * w + x] = (0xFF shl 24) or (r.coerceIn(0, 255) shl 16) or (g.coerceIn(0, 255) shl 8) or b.coerceIn(0, 255)
            }
        }
        val result = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        result.setPixels(outPx, 0, w, 0, 0, w, h)
        return result
    }

    private fun outputToBitmap(output: Array<Array<Array<FloatArray>>>, outputSize: Int): Bitmap {
        val c0 = output[0][0]
        val c1 = output[0][1]
        val c2 = output[0][2]
        val px = IntArray(outputSize * outputSize)
        for (y in 0 until outputSize) {
            for (x in 0 until outputSize) {
                val b = (c0[y][x].coerceIn(0f, 1f) * 255f).toInt().coerceIn(0, 255)
                val g = (c1[y][x].coerceIn(0f, 1f) * 255f).toInt().coerceIn(0, 255)
                val r = (c2[y][x].coerceIn(0f, 1f) * 255f).toInt().coerceIn(0, 255)
                px[y * outputSize + x] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
        }
        val bmp = Bitmap.createBitmap(outputSize, outputSize, Bitmap.Config.ARGB_8888)
        bmp.setPixels(px, 0, outputSize, 0, 0, outputSize, outputSize)
        return if (outputSize == PATCH_160) bmp else Bitmap.createScaledBitmap(bmp, PATCH_160, PATCH_160, true).also { bmp.recycle() }
    }

    private class VideoEncoder(
        private val width: Int,
        private val height: Int,
        private val fps: Int
    ) {
        private var codec: MediaCodec? = null
        private var muxer: MediaMuxer? = null
        private var trackIndex = -1
        private var started = false
        private var frameCount = 0L

        fun start(outputPath: String) {
            // 服务端用 MJPEG（每帧独立 JPEG），画质好；端上 H264 有块效应。提高码率减轻方框感
            val bitrate = (width * height * 8).coerceAtLeast(8_000_000)
            val format = MediaFormat.createVideoFormat(MediaFormat.MIMETYPE_VIDEO_AVC, width, height).apply {
                setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420SemiPlanar)
                setInteger(MediaFormat.KEY_BIT_RATE, bitrate)
                setInteger(MediaFormat.KEY_FRAME_RATE, fps)
                setFloat(MediaFormat.KEY_I_FRAME_INTERVAL, 0.5f)
            }
            codec = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_VIDEO_AVC).apply {
                configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
                start()
            }
            muxer = MediaMuxer(outputPath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
        }

        fun encodeFrame(bitmap: Bitmap) {
            val c = codec ?: return
            val yuv = bitmapToNV12(bitmap)
            val bufferInfo = MediaCodec.BufferInfo()
            var inputIndex = c.dequeueInputBuffer(10000)
            if (inputIndex >= 0) {
                val inputBuffer = c.getInputBuffer(inputIndex) ?: return
                inputBuffer.clear()
                inputBuffer.put(yuv)
                val pts = frameCount * 1_000_000L / fps
                c.queueInputBuffer(inputIndex, 0, yuv.size, pts, 0)
                frameCount++
            }
            var outputIndex = c.dequeueOutputBuffer(bufferInfo, 10000)
            while (outputIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                val m = muxer ?: break
                if (!started) {
                    trackIndex = m.addTrack(c.outputFormat)
                    m.start()
                    started = true
                }
                outputIndex = c.dequeueOutputBuffer(bufferInfo, 10000)
            }
            if (outputIndex >= 0 && started) {
                val outputBuffer = c.getOutputBuffer(outputIndex) ?: return
                muxer?.writeSampleData(trackIndex, outputBuffer, bufferInfo)
                c.releaseOutputBuffer(outputIndex, false)
            }
        }

        fun stop() {
            val c = codec ?: return
            val bufferInfo = MediaCodec.BufferInfo()
            try {
                try {
                    c.signalEndOfInputStream()
                } catch (e: IllegalStateException) {
                    Log.w(LOG_TAG, "signalEndOfInputStream ignored (MTK/device quirk): ${e.message}")
                }
                while (true) {
                    val idx = c.dequeueOutputBuffer(bufferInfo, 10000)
                    when {
                        idx == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                            if (!started) {
                                trackIndex = muxer!!.addTrack(c.outputFormat)
                                muxer!!.start()
                                started = true
                            }
                        }
                        idx >= 0 -> {
                            val buf = c.getOutputBuffer(idx) ?: break
                            if (started && (bufferInfo.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG) == 0) {
                                muxer?.writeSampleData(trackIndex, buf, bufferInfo)
                            }
                            c.releaseOutputBuffer(idx, false)
                            if ((bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) break
                        }
                        else -> break
                    }
                }
            } finally {
                c.stop()
                c.release()
                muxer?.stop()
                muxer?.release()
                codec = null
                muxer = null
            }
        }

        private fun bitmapToNV12(bitmap: Bitmap): ByteArray {
            val w = bitmap.width
            val h = bitmap.height
            val ySize = w * h
            val uvSize = ySize / 2
            val nv12 = ByteArray(ySize + uvSize)
            val pixels = IntArray(w * h)
            bitmap.getPixels(pixels, 0, w, 0, 0, w, h)
            var yIdx = 0
            var uvIdx = ySize
            for (j in 0 until h) {
                for (i in 0 until w) {
                    val c = pixels[j * w + i]
                    val r = (c shr 16) and 0xFF
                    val g = (c shr 8) and 0xFF
                    val b = c and 0xFF
                    val y = ((66 * r + 129 * g + 25 * b + 128) shr 8) + 16
                    nv12[yIdx++] = y.coerceIn(0, 255).toByte()
                }
                if (j and 1 == 0) {
                    for (i in 0 until w step 2) {
                        val c1 = pixels[j * w + i]
                        val c2 = if (j + 1 < h) pixels[(j + 1) * w + i] else c1
                        val r = ((c1 shr 16) and 0xFF + (c2 shr 16) and 0xFF) / 2
                        val g = ((c1 shr 8) and 0xFF + (c2 shr 8) and 0xFF) / 2
                        val b = (c1 and 0xFF + c2 and 0xFF) / 2
                        val u = ((-38 * r - 74 * g + 112 * b + 128) shr 8) + 128
                        val v = ((112 * r - 94 * g - 18 * b + 128) shr 8) + 128
                        nv12[uvIdx++] = u.coerceIn(0, 255).toByte()
                        nv12[uvIdx++] = v.coerceIn(0, 255).toByte()
                    }
                }
            }
            return nv12
        }
    }
}
