/**
 * @file Preprocessor.cpp
 * @brief 预处理实现
 */

#include "Preprocessor.hpp"
#include <cstring>
#include <algorithm>

namespace digital_human {

Preprocessor::Preprocessor(int32_t target_width, int32_t target_height)
    : target_width_(target_width), target_height_(target_height) {}

void Preprocessor::processImage(const float* input, int32_t width, int32_t height,
                                 float* output) {
    // 调整大小
    std::vector<float> resized(width * height * 3);
    resizeBilinear(input, width, height, resized.data(), target_width_, target_height_);
    
    // 归一化到[0, 1]
    int32_t total_pixels = target_width_ * target_height_;
    for (int32_t i = 0; i < total_pixels * 3; ++i) {
        output[i] = resized[i] / 255.0f;
    }
    
    // ImageNet归一化
    normalize(output, total_pixels * 3, kMean, kStd);
    
    // HWC -> CHW
    for (int32_t y = 0; y < target_height_; ++y) {
        for (int32_t x = 0; x < target_width_; ++x) {
            for (int32_t c = 0; c < 3; ++c) {
                int32_t hwc_idx = (y * target_width_ + x) * 3 + c;
                int32_t chw_idx = c * target_width_ * target_height_ + y * target_width_ + x;
                output[chw_idx] = output[hwc_idx];
            }
        }
    }
}

void Preprocessor::processAudio(int32_t num_frames, int32_t feature_dim, 
                                 float* output) {
    // 音频特征已经在Java层处理
    // 这里只是简单复制
    // 实际使用中可能需要进一步处理
}

void Preprocessor::resizeBilinear(const float* input, int32_t src_width, int32_t src_height,
                                  float* output, int32_t dst_width, int32_t dst_height) {
    float scale_x = static_cast<float>(src_width) / dst_width;
    float scale_y = static_cast<float>(src_height) / dst_height;
    
    for (int32_t dy = 0; dy < dst_height; ++dy) {
        float src_y = dy * scale_y;
        int32_t y0 = static_cast<int32_t>(src_y);
        int32_t y1 = std::min(y0 + 1, src_height - 1);
        float vy = src_y - y0;
        
        for (int32_t dx = 0; dx < dst_width; ++dx) {
            float src_x = dx * scale_x;
            int32_t x0 = static_cast<int32_t>(src_x);
            int32_t x1 = std::min(x0 + 1, src_width - 1);
            float vx = src_x - x0;
            
            for (int32_t c = 0; c < 3; ++c) {
                // 双线性插值
                float p00 = input[(y0 * src_width + x0) * 3 + c];
                float p10 = input[(y0 * src_width + x1) * 3 + c];
                float p01 = input[(y1 * src_width + x0) * 3 + c];
                float p11 = input[(y1 * src_width + x1) * 3 + c];
                
                float value = (1 - vx) * (1 - vy) * p00 +
                              vx * (1 - vy) * p10 +
                              (1 - vx) * vy * p01 +
                              vx * vy * p11;
                
                output[(dy * dst_width + dx) * 3 + c] = value;
            }
        }
    }
}

void Preprocessor::normalize(float* data, int32_t size,
                              const float* mean, const float* std) {
    for (int32_t i = 0; i < size; ++i) {
        int32_t c = i % 3;
        data[i] = (data[i] - mean[c]) / std[c];
    }
}

void Preprocessor::convertBGRtoRGB(float* data, int32_t size) {
    int32_t num_pixels = size / 3;
    for (int32_t i = 0; i < num_pixels; ++i) {
        float tmp = data[i * 3];
        data[i * 3] = data[i * 3 + 2];
        data[i * 3 + 2] = tmp;
    }
}

} // namespace digital_human
