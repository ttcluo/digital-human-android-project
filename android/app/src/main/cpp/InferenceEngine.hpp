/**
 * @file InferenceEngine.hpp
 * @brief 数字人推理引擎头文件
 */

#ifndef DIGITAL_HUMAN_INFERENCE_ENGINE_HPP
#define DIGITAL_HUMAN_INFERENCE_ENGINE_HPP

#include <jni.h>
#include <android/asset_manager.h>
#include <string>
#include <vector>
#include <memory>
#include <mutex>

namespace digital_human {

/**
 * @brief 推理引擎配置
 */
struct EngineConfig {
    int num_threads = 4;
    bool use_gpu = false;
    int32_t input_width = 256;
    int32_t input_height = 256;
    int32_t audio_feature_dim = 256;
    bool enable_xnnpack = true;
};

/**
 * @brief 推理结果
 */
struct InferenceResult {
    std::vector<float> output_data;
    int32_t width;
    int32_t height;
    int32_t channels;
    float inference_time_ms;
};

/**
 * @brief 数字人推理引擎
 */
class InferenceEngine {
public:
    InferenceEngine();
    ~InferenceEngine();

    /**
     * @brief 初始化引擎
     * @param model_path 模型文件路径
     * @param config 引擎配置
     * @return 是否初始化成功
     */
    bool initialize(const std::string& model_path, const EngineConfig& config);

    /**
     * @brief 执行推理
     * @param input_image 输入图像数据 [H, W, C] RGB格式
     * @param audio_features 音频特征 [T, D]
     * @return 推理结果
     */
    InferenceResult infer(const float* input_image, 
                         const float* audio_features,
                         int32_t image_width,
                         int32_t image_height,
                         int32_t audio_frames);

    /**
     * @brief 流式推理
     * @param input_image 输入图像
     * @param audio_feature 单帧音频特征
     * @return 推理结果
     */
    InferenceResult inferStream(const float* input_image,
                               const float* audio_feature,
                               int32_t image_width,
                               int32_t image_height);

    /**
     * @brief 获取模型信息
     */
    void getModelInfo(int32_t& input_width, int32_t& input_height, 
                     int32_t& num_parameters) const;

    /**
     * @brief 释放资源
     */
    void release();

    /**
     * @brief 是否可用
     */
    bool isReady() const { return is_ready_; }

private:
    // ONNX Runtime会话
    void* session_;  // OrtSession*

    // 配置
    EngineConfig config_;

    // 状态
    bool is_ready_ = false;
    bool use_gpu_ = false;

    // 线程安全
    std::mutex inference_mutex_;

    // 预分配的缓冲区
    std::vector<float> input_buffer_;
    std::vector<float> output_buffer_;

    // 初始化ONNX Runtime
    bool initOnnxRuntime(const std::string& model_path);

    // 预处理输入
    void preprocess(const float* input_image, 
                   int32_t width, int32_t height,
                   int32_t audio_frames);

    // 后处理输出
    InferenceResult postprocess(float* output_data);
};

/**
 * @brief 推理引擎工厂
 */
class InferenceEngineFactory {
public:
    static std::unique_ptr<InferenceEngine> create();
};

} // namespace digital_human

#endif // DIGITAL_HUMAN_INFERENCE_ENGINE_HPP
