#include <jni.h>
#include <android/log.h>
#include <chrono>
#include <string>

#define LOG_TAG "UnetBenchmark"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

extern "C"
JNIEXPORT jfloat JNICALL
Java_com_digitalhuman_app_UnetBenchmarkActivity_nativeDummyBenchmark(
        JNIEnv *env,
        jobject /* this */,
        jint iters) {
    int iterations = iters > 0 ? iters : 100;
    auto t0 = std::chrono::high_resolution_clock::now();
    volatile float x = 0.f;
    for (int i = 0; i < iterations * 100000; ++i) {
        x += 1.f;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    float ms = std::chrono::duration<float, std::milli>(t1 - t0).count() / iterations;
    LOGI("dummy benchmark avg_ms=%.4f", ms);
    return ms;
}