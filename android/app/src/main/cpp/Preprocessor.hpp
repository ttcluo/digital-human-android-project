/**
 * @file Preprocessor.hpp
 * @brief 输入预处理模块
 */

#ifndef DIGITAL_HUMAN_PREPROCESSOR_HPP
#define DIGITAL_HUMAN_PREPROCESSOR_HPP

#include <cstdint>
#include <vector>

namespace digital_human {

/**
 * @brief 图像和音频预处理
 */
class Preprocessor {
public:
    Preprocessor(int32_t target_width, int32_t target_height);
    
    /**
     * @brief 预处理图像
     * @param input 输入图像数据 [H, W, C] RGB格式, 0-255
     * @param width 输入宽度
     * @param height 输入高度
     * @param output 输出缓冲区
     */
    void processImage(const float* input, int32_t width, int32_t height,
                     float* output);
    
    /**
     * @brief 预处理音频特征
     * @param num_frames 帧数
     * @param feature_dim 特征维度
     * @param output 输出缓冲区
     */
    void processAudio(int32_t num_frames, int32_t feature_dim, float* output);
    
    /**
     * @brief 调整图像大小（双线性插值）
     */
    static void resizeBilinear(const float* input, int32_t src_width, int32_t src_height,
                               float* output, int32_t dst_width, int32_t dst_height);
    
    /**
     * @brief 归一化图像
     */
    static void normalize(float* data, int32_t size, 
                         const float* mean, const float* std);
    
    /**
     * @brief 图像色彩空间转换 BGR -> RGB
     */
    static void convertBGRtoRGB(float* data, int32_t size);

private:
    int32_t target_width_;
    int32_t target_height_;
    
    // ImageNet统计
    static constexpr float kMean[3] = {0.485f, 0.456f, 0.406f};
    static constexpr float kStd[3] = {0.229f, 0.224f, 0.225f};
};

} // namespace digital_human

#endif // DIGITAL_HUMAN_PREPROCESSOR_HPP
