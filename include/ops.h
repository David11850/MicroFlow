#ifndef OPS_H
#define OPS_H

#include "tensor.h"

namespace microflow {

// 基准实现
void gemm_naive(const Tensor& A, const Tensor& B, Tensor& C);

// Level 1: OpenMP 多线程优化
void gemm_omp(const Tensor& A, const Tensor& B, Tensor& C);

// Level 2: AVX2 SIMD (x86 专用优化)
void gemm_avx(const Tensor& A, const Tensor& B, Tensor& C);


// 卷积层: Im2Col + GEMM
// input: [C, H, W]
// kernel: [FN, C, K, K] (FN: 卷积核个数)
// output: [FN, OH, OW]
void conv2d(const Tensor& input, const Tensor& kernel, Tensor& output, 
           uint32_t stride, uint32_t padding);

// 激活函数: ReLU
// In-place 操作：直接修改输入 Tensor
void relu(Tensor& tensor);

// 池化层: MaxPool2d
// 目前仅支持 stride = kernel_size 的情况 (最常用)
// input: [C, H, W]
// output: [C, H/k, W/k]
void max_pool2d(const Tensor& input, Tensor& output, uint32_t kernel_size);

// 全连接层 (Linear Layer)
// Y = X * W^T + B
// input: [M, K]
// weight: [N, K]
// bias: [N] (可选)
// output: [M, N]
void linear(const Tensor& input, const Tensor& weight, const Tensor& bias, Tensor& output);

} // namespace microflow

#endif // OPS_H