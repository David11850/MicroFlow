#include "im2col.h"
#include <cmath>

namespace microflow {

void im2col(const Tensor& input, Tensor& output, 
            uint32_t k_size, uint32_t stride, uint32_t padding) {
    
    // 1. 获取输入维度
    // 假设 input shape 是 [C, H, W] (暂时忽略 Batch 维度，简化模型)
    uint32_t C = input.shapes()[0];
    uint32_t H = input.shapes()[1];
    uint32_t W = input.shapes()[2];

    // 2. 计算输出特征图的大小
    // Formula: (H + 2*pad - k) / stride + 1
    uint32_t out_h = (H + 2 * padding - k_size) / stride + 1;
    uint32_t out_w = (W + 2 * padding - k_size) / stride + 1;

    // 3. 检查 output 张量形状是否正确
    // im2col 的结果矩阵行数 = 卷积核体积 = C * k * k
    // im2col 的结果矩阵列数 = 滑动窗口总数 = out_h * out_w
    uint32_t matrix_rows = C * k_size * k_size;
    uint32_t matrix_cols = out_h * out_w;

    if (output.size() != matrix_rows * matrix_cols) {
        std::cerr << "Im2Col Error: Output tensor size mismatch!" << std::endl;
        return;
    }

    const float* src = input.raw_ptr();
    float* dst = output.raw_ptr();

    // 4. 核心循环：填充 Im2Col 矩阵
    // 我们遍历输出矩阵的每一列（对应原始图片的一个滑动窗口）
    
    // 优化提示：这里可以用 OpenMP，但为了逻辑清晰，我们先写单线程
    for (uint32_t c = 0; c < matrix_cols; ++c) {
        // 计算当前窗口在输出特征图中的坐标 (w_out, h_out)
        uint32_t w_out = c % out_w;
        uint32_t h_out = c / out_w;

        // 计算对应输入图片中的左上角坐标 (h_in, w_in)
        // 记得考虑 padding，所以这里可能是负数，用 int 而不是 uint
        int h_in_start = h_out * stride - padding;
        int w_in_start = w_out * stride - padding;

        // 遍历卷积核体积 (C * k * k)，填充这一列
        for (uint32_t k_ch = 0; k_ch < C; ++k_ch) {
            for (uint32_t k_y = 0; k_y < k_size; ++k_y) {
                for (uint32_t k_x = 0; k_x < k_size; ++k_x) {
                    
                    int h_in = h_in_start + k_y;
                    int w_in = w_in_start + k_x;
                    
                    // 计算 dst 指针位置：Row-Major
                    // dst_row = (channel * k * k) + (k_y * k) + k_x
                    // dst_col = c
                    uint32_t dst_idx = ((k_ch * k_size * k_size + k_y * k_size + k_x) * matrix_cols) + c;

                    // Padding 处理：如果在图片外，填 0
                    if (h_in >= 0 && h_in < (int)H && w_in >= 0 && w_in < (int)W) {
                        // 对应的 src 索引
                        uint32_t src_idx = (k_ch * H * W) + (h_in * W) + w_in;
                        dst[dst_idx] = src[src_idx];
                    } else {
                        dst[dst_idx] = 0.0f;
                    }
                }
            }
        }
    }
}

} // namespace microflow