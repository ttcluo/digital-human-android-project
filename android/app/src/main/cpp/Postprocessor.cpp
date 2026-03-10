/**
 * @file Postprocessor.cpp
 * @brief 后处理实现
 */

#include "Postprocessor.hpp"
#include <cmath>
#include <algorithm>

namespace digital_human {

Postprocessor::Postprocessor() {}

void Postprocessor::processOutput(const float* output, float* result,
                                  int32_t width, int32_t height) {
    int32_t num_pixels = width * height;
    
    // CHW -> HWC
    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            for (int32_t c = 0; c < 3; ++c) {
                int32_t chw_idx = c * num_pixels + y * width + x;
                int32_t hwc_idx = (y * width + x) * 3 + c;
                result[hwc_idx] = output[chw_idx];
            }
        }
    }
    
    // 反归一化
    denormalize(result, num_pixels * 3, kMean, kStd);
    
    // 裁剪到[0, 1]
    clip(result, num_pixels * 3, 0.0f, 1.0f);
}

void Postprocessor::denormalize(float* data, int32_t size,
                                const float* mean, const float* std) {
    for (int32_t i = 0; i < size; ++i) {
        int32_t c = i % 3;
        data[i] = data[i] * std[c] + mean[c];
    }
}

void Postprocessor::clip(float* data, int32_t size, float min_val, float max_val) {
    for (int32_t i = 0; i < size; ++i) {
        data[i] = std::max(min_val, std::min(max_val, data[i]));
    }
}

void Postprocessor::convertToUint8(const float* input, uint8_t* output, int32_t size) {
    for (int32_t i = 0; i < size; ++i) {
        output[i] = static_cast<uint8_t>(std::round(input[i] * 255.0f));
    }
}

} // namespace digital_human
