/**
 * @file Postprocessor.hpp
 * @brief 输出后处理模块
 */

#ifndef DIGITAL_HUMAN_POSTPROCESSOR_HPP
#define DIGITAL_HUMAN_POSTPROCESSOR_HPP

#include <cstdint>
#include <vector>

namespace digital_human {

/**
 * @brief 输出后处理
 */
class Postprocessor {
public:
    Postprocessor();
    
    /**
     * @brief 处理模型输出
     * @param output 模型输出 [C, H, W]
     * @param result 最终结果 [H, W, C] RGB格式, 0-255
     */
    void processOutput(const float* output, float* result,
                      int32_t width, int32_t height);
    
    /**
     * @brief 反归一化
     */
    static void denormalize(float* data, int32_t size,
                           const float* mean, const float* std);
    
    /**
     * @brief 裁剪到有效范围
     */
    static void clip(float* data, int32_t size, float min_val, float max_val);
    
    /**
     * @brief 转换为uint8
     */
    static void convertToUint8(const float* input, uint8_t* output, int32_t size);

private:
    // ImageNet统计
    static constexpr float kMean[3] = {0.485f, 0.456f, 0.406f};
    static constexpr float kStd[3] = {0.229f, 0.224f, 0.225f};
};

} // namespace digital_human

#endif // DIGITAL_HUMAN_POSTPROCESSOR_HPP
