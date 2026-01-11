#ifndef IM2COL_H
#define IM2COL_H

#include "tensor.h"

namespace microflow {

// 将 3D 图片张量展开为 2D 矩阵，以便进行 GEMM
// Input Tensor: [C, H, W]
// Output Tensor: [K*K*C, Output_H * Output_W]
// k_size: 卷积核大小 (假设长宽相等)
// stride: 步长
// padding: 填充
void im2col(const Tensor& input, Tensor& output, 
            uint32_t k_size, uint32_t stride, uint32_t padding);

} // namespace microflow

#endif // IM2COL_H