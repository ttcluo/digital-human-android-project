/**
 * @file digital_human_jni.cpp
 * @brief JNI接口 - Java/Kotlin与C++的桥接
 */

#include <jni.h>
#include <android/bitmap.h>
#include <android/log.h>
#include <string>
#include <memory>
#include <vector>

#include "InferenceEngine.hpp"
#include "Preprocessor.hpp"
#include "Postprocessor.hpp"

#define LOG_TAG "DigitalHuman JNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace digital_human {
    static std::unique_ptr<InferenceEngine> g_engine;
}

extern "C" {

/**
 * 初始化推理引擎
 */
JNIEXPORT jboolean JNICALL
Java_com_digitalhuman_app_engine_DigitalHumanEngine_nativeInitialize(
        JNIEnv* env, jobject thiz,
        jstring model_path,
        jint num_threads,
        jboolean use_gpu,
        jint input_size) {
    
    // 获取模型路径
    const char* model_path_str = env->GetStringUTFChars(model_path, nullptr);
    std::string path(model_path_str);
    env->ReleaseStringUTFChars(model_path, model_path_str);
    
    // 配置引擎
    digital_human::EngineConfig config;
    config.num_threads = num_threads;
    config.use_gpu = use_gpu;
    config.input_width = input_size;
    config.input_height = input_size;
    config.audio_feature_dim = 256;
    config.enable_xnnpack = true;
    
    // 创建并初始化引擎
    digital_human::g_engine = digital_human::InferenceEngineFactory::create();
    bool success = digital_human::g_engine->initialize(path, config);
    
    LOGI("Engine initialized: %s", success ? "success" : "failed");
    return success ? JNI_TRUE : JNI_FALSE;
}

/**
 * 执行推理
 */
JNIEXPORT jboolean JNICALL
Java_com_digitalhuman_app_engine_DigitalHumanEngine_nativeInfer(
        JNIEnv* env, jobject thiz,
        jobject bitmap_input,
        jfloatArray audio_features,
        jobject bitmap_output) {
    
    if (!digital_human::g_engine || !digital_human::g_engine->isReady()) {
        LOGE("Engine not ready");
        return JNI_FALSE;
    }
    
    // 获取输入图像
    AndroidBitmapInfo input_info;
    void* input_pixels = nullptr;
    
    if (AndroidBitmap_getInfo(env, bitmap_input, &input_info) < 0) {
        LOGE("Failed to get input bitmap info");
        return JNI_FALSE;
    }
    
    if (AndroidBitmap_lockPixels(env, bitmap_input, &input_pixels) < 0) {
        LOGE("Failed to lock input bitmap");
        return JNI_FALSE;
    }
    
    // 获取音频特征
    jfloat* audio_data = env->GetFloatArrayElements(audio_features, nullptr);
    jsize audio_size = env->GetArrayLength(audio_features);
    
    // 获取输出图像
    AndroidBitmapInfo output_info;
    void* output_pixels = nullptr;
    
    if (AndroidBitmap_getInfo(env, bitmap_output, &output_info) < 0) {
        LOGE("Failed to get output bitmap info");
        AndroidBitmap_unlockPixels(env, bitmap_input);
        env->ReleaseFloatArrayElements(audio_features, audio_data, 0);
        return JNI_FALSE;
    }
    
    if (AndroidBitmap_lockPixels(env, bitmap_output, &output_pixels) < 0) {
        LOGE("Failed to lock output bitmap");
        AndroidBitmap_unlockPixels(env, bitmap_input);
        env->ReleaseFloatArrayElements(audio_features, audio_data, 0);
        return JNI_FALSE;
    }
    
    // 准备输入数据
    int32_t width = input_info.width;
    int32_t height = input_info.height;
    std::vector<float> input_image(width * height * 3);
    
    // RGBA转RGB (假设输入是RGBA_8888)
    uint32_t* pixels = static_cast<uint32_t*>(input_pixels);
    for (int32_t i = 0; i < width * height; ++i) {
        uint32_t pixel = pixels[i];
        input_image[i * 3 + 0] = (pixel >> 16) & 0xFF; // R
        input_image[i * 3 + 1] = (pixel >> 8) & 0xFF;   // G
        input_image[i * 3 + 2] = pixel & 0xFF;           // B
    }
    
    // 执行推理
    auto result = digital_human::g_engine->infer(
        input_image.data(),
        audio_data,
        width,
        height,
        audio_size / 256
    );
    
    // 将输出转换为Bitmap
    uint32_t* out_pixels = static_cast<uint32_t*>(output_pixels);
    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            int32_t idx = (y * width + x) * 3;
            uint8_t r = static_cast<uint8_t>(std::round(result.output_data[idx + 0] * 255.0f));
            uint8_t g = static_cast<uint8_t>(std::round(result.output_data[idx + 1] * 255.0f));
            uint8_t b = static_cast<uint8_t>(std::round(result.output_data[idx + 2] * 255.0f));
            out_pixels[y * width + x] = (0xFF << 24) | (r << 16) | (g << 8) | b;
        }
    }
    
    // 解锁
    AndroidBitmap_unlockPixels(env, bitmap_input);
    AndroidBitmap_unlockPixels(env, bitmap_output);
    env->ReleaseFloatArrayElements(audio_features, audio_data, 0);
    
    LOGI("Inference completed in %.2f ms", result.inference_time_ms);
    return JNI_TRUE;
}

/**
 * 释放资源
 */
JNIEXPORT void JNICALL
Java_com_digitalhuman_app_engine_DigitalHumanEngine_nativeRelease(
        JNIEnv* env, jobject thiz) {
    
    if (digital_human::g_engine) {
        digital_human::g_engine->release();
        digital_human::g_engine.reset();
    }
    
    LOGI("Engine released");
}

/**
 * 获取模型信息
 */
JNIEXPORT void JNICALL
Java_com_digitalhuman_app_engine_DigitalHumanEngine_nativeGetModelInfo(
        JNIEnv* env, jobject thiz,
        jintArray model_info) {
    
    if (!digital_human::g_engine) {
        return;
    }
    
    int32_t width, height, num_params;
    digital_human::g_engine->getModelInfo(width, height, num_params);
    
    jint* info = env->GetIntArrayElements(model_info, nullptr);
    info[0] = width;
    info[1] = height;
    info[2] = num_params;
    env->ReleaseIntArrayElements(model_info, info, 0);
}

} // extern "C"
