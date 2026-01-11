#include "ops.h"
#include <cassert>
#include <omp.h>
#include "im2col.h"
#include <algorithm> // for std::min

// ---------------- HARDWARE ABSTRACTION LAYER ----------------
// 只有在 x86 架构下，才引入 AVX 头文件
#if defined(__x86_64__) || defined(_M_X64)
    #include <immintrin.h>
    #define ENABLE_AVX2 // 标记开启 AVX2
#endif

// 只有在 ARM 架构下，才引入 NEON 头文件 (为未来做准备)
#if defined(__aarch64__) || defined(_M_ARM64)
    #include <arm_neon.h>
    #define ENABLE_NEON // 标记开启 NEON
#endif
// ------------------------------------------------------------

namespace microflow {

// ==========================================
//           Common Implementations
// ==========================================

void gemm_naive(const Tensor& A, const Tensor& B, Tensor& C) {
    uint32_t M = A.shapes()[0];
    uint32_t K = A.shapes()[1];
    uint32_t N = B.shapes()[1];

    assert(A.shapes()[1] == B.shapes()[0]); 
    assert(C.shapes()[0] == M);
    assert(C.shapes()[1] == N);

    const float* ptr_A = A.raw_ptr();
    const float* ptr_B = B.raw_ptr();
    float* ptr_C = C.raw_ptr();

    for (uint32_t i = 0; i < M; ++i) {            
        for (uint32_t j = 0; j < N; ++j) {        
            float sum = 0.0f;
            for (uint32_t k = 0; k < K; ++k) {    
                uint32_t offset_A = i * A.strides()[0] + k * A.strides()[1];
                uint32_t offset_B = k * B.strides()[0] + j * B.strides()[1];
                sum += ptr_A[offset_A] * ptr_B[offset_B];
            }
            uint32_t offset_C = i * C.strides()[0] + j * C.strides()[1];
            ptr_C[offset_C] = sum;
        }
    }
}

void gemm_omp(const Tensor& A, const Tensor& B, Tensor& C) {
    uint32_t M = A.shapes()[0];
    uint32_t K = A.shapes()[1];
    uint32_t N = B.shapes()[1];

    const float* ptr_A = A.raw_ptr();
    const float* ptr_B = B.raw_ptr();
    float* ptr_C = C.raw_ptr();

    #pragma omp parallel for
    for (uint32_t i = 0; i < M; ++i) {            
        for (uint32_t j = 0; j < N; ++j) {        
            float sum = 0.0f;
            for (uint32_t k = 0; k < K; ++k) {    
                uint32_t offset_A = i * A.strides()[0] + k * A.strides()[1];
                uint32_t offset_B = k * B.strides()[0] + j * B.strides()[1];
                sum += ptr_A[offset_A] * ptr_B[offset_B];
            }
            uint32_t offset_C = i * C.strides()[0] + j * C.strides()[1];
            ptr_C[offset_C] = sum;
        }
    }
}

// [修改实现]
void conv2d(const Tensor& input, const Tensor& kernel, Tensor& output, 
            uint32_t stride, uint32_t padding, float* workspace) {
    
    uint32_t H = input.shapes()[1]; 
    uint32_t W = input.shapes()[2];
    uint32_t FN = kernel.shapes()[0];
    uint32_t C  = kernel.shapes()[1];
    uint32_t K  = kernel.shapes()[2];
    uint32_t OH = (H + 2 * padding - K) / stride + 1;
    uint32_t OW = (W + 2 * padding - K) / stride + 1;

    // -------------------------------------------------------
    // [核心修改点]
    // -------------------------------------------------------
    
    // 检查是否传入了 workspace
    if (workspace == nullptr) {
        // 如果用户没传，为了兼容性，我们还是 fallback 到原来的 malloc 模式 (但在高性能模式下不应发生)
        // 这里可以打印一个警告
        std::cerr << "[Performance Warning] conv2d called without workspace! Triggering malloc." << std::endl;
        Tensor col_buffer({C * K * K, OH * OW}); 
        im2col(input, col_buffer, K, stride, padding);
        // ... (后续逻辑需要调整，稍微麻烦，建议强制要求 workspace)
        // 为了简化，我们直接假设 workspace 必须存在
        std::exit(-1); 
    }

    // 1. 创建 Im2Col 缓冲区 (View Mode)
    // 这一步不会分配内存，只是创建了一个指向 workspace 的 Tensor 头信息
    Tensor col_buffer({C * K * K, OH * OW}, workspace); 

    // 2. 执行 Im2Col (数据被写到了 workspace 里)
    im2col(input, col_buffer, K, stride, padding);

    // 3. 准备输出矩阵的 View
    // 注意：gemm 的输出也不应该 malloc，但为了简单起见，我们暂时认为 output 是外部传入的 Tensor
    // 这里主要优化的是 col_buffer 这个巨大的中间变量

    // 4. Kernel 展平
    // 这一步其实也可以预处理 (Model Loader阶段完成)，暂时先保留拷贝
    Tensor kernel_mat({FN, C*K*K});
    std::copy(kernel.raw_ptr(), kernel.raw_ptr() + kernel.size(), kernel_mat.raw_ptr());
    
    Tensor output_mat({FN, OH * OW});

    // 5. GEMM
    gemm_omp(kernel_mat, col_buffer, output_mat);

    // 6. 拷贝回结果
    std::copy(output_mat.raw_ptr(), output_mat.raw_ptr() + output_mat.size(), output.raw_ptr());
}

// ==========================================
//           x86 AVX2 Implementation
// ==========================================
// 关键点：这里加了 #ifdef ENABLE_AVX2
// 如果没有这个宏（比如在 ARM 上），这部分代码会被编译器直接忽略，从而避免报错
#ifdef ENABLE_AVX2 

#define BLOCK_SIZE 64

void gemm_micro_kernel(int M, int N, int K, 
                       const float* A, int lda, 
                       const float* B, int ldb, 
                       float* C, int ldc) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; j += 8) { 
            __m256 sum_vec = _mm256_loadu_ps(&C[i * ldc + j]); 
            for (int k = 0; k < K; ++k) {
                __m256 vec_A = _mm256_set1_ps(A[i * lda + k]);
                __m256 vec_B = _mm256_loadu_ps(&B[k * ldb + j]);
#ifdef __FMA__
                sum_vec = _mm256_fmadd_ps(vec_A, vec_B, sum_vec);
#else
                sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(vec_A, vec_B));
#endif
            }
            _mm256_storeu_ps(&C[i * ldc + j], sum_vec);
        }
    }
}

void gemm_avx(const Tensor& A, const Tensor& B, Tensor& C) {
    uint32_t M = A.shapes()[0];
    uint32_t K = A.shapes()[1];
    uint32_t N = B.shapes()[1];
    const float* ptr_A = A.raw_ptr();
    const float* ptr_B = B.raw_ptr();
    float* ptr_C = C.raw_ptr();

    #pragma omp parallel for
    for (uint32_t i = 0; i < M; i += BLOCK_SIZE) {
        for (uint32_t k = 0; k < K; k += BLOCK_SIZE) {
            for (uint32_t j = 0; j < N; j += BLOCK_SIZE) {
                int M_cur = std::min((uint32_t)BLOCK_SIZE, M - i);
                int K_cur = std::min((uint32_t)BLOCK_SIZE, K - k);
                int N_cur = std::min((uint32_t)BLOCK_SIZE, N - j);
                const float* A_ptr = ptr_A + i * K + k;
                const float* B_ptr = ptr_B + k * N + j;
                float* C_ptr = ptr_C + i * N + j;
                gemm_micro_kernel(M_cur, N_cur, K_cur, A_ptr, K, B_ptr, N, C_ptr, N);
            }
        }
    }
}

#else 

// 这是一个“桩” (Stub)
// 如果我们在不支持 AVX2 的机器上调用了 gemm_avx，它会报错而不是让程序崩溃
#include <iostream>
void gemm_avx(const Tensor& A, const Tensor& B, Tensor& C) {
    std::cerr << "[Fatal Error] AVX2 not supported on this architecture!" << std::endl;
    std::exit(-1);
}

#endif // ENABLE_AVX2


// ==========================================
//           ARM NEON Implementation
// ==========================================
#ifdef ENABLE_NEON

// 预留给未来的 ARM 实现
void gemm_neon(const Tensor& A, const Tensor& B, Tensor& C) {
    // Future Work: float32x4_t implementation
}

#else

// 占位符
void gemm_neon(const Tensor& A, const Tensor& B, Tensor& C) {
    std::cerr << "[Warning] NEON not supported on this architecture." << std::endl;
}

#endif // ENABLE_NEON

// ==========================================
//           Activation & Pooling
// ==========================================

void relu(Tensor& tensor) {
    uint32_t size = tensor.size();
    float* ptr = tensor.raw_ptr();

    // SIMD 优化版 ReLU
#ifdef ENABLE_AVX2
    // 每次处理 8 个 float
    uint32_t i = 0;
    __m256 zero = _mm256_setzero_ps(); // 准备一个全 0 的向量
    
    #pragma omp parallel for
    for (i = 0; i <= size - 8; i += 8) {
        // 加载数据
        __m256 data = _mm256_loadu_ps(ptr + i);
        // max(0, data)
        __m256 res = _mm256_max_ps(zero, data);
        // 存回
        _mm256_storeu_ps(ptr + i, res);
    }
    // 处理剩余的尾巴
    for (; i < size; ++i) {
        ptr[i] = std::max(0.0f, ptr[i]);
    }

#else
    // 普通版 (OpenMP 加速)
    #pragma omp parallel for
    for (uint32_t i = 0; i < size; ++i) {
        ptr[i] = std::max(0.0f, ptr[i]);
    }
#endif
}

void max_pool2d(const Tensor& input, Tensor& output, uint32_t k_size) {
    uint32_t C = input.shapes()[0];
    uint32_t H = input.shapes()[1];
    uint32_t W = input.shapes()[2];
    
    // 输出尺寸
    uint32_t OH = output.shapes()[1];
    uint32_t OW = output.shapes()[2];

    const float* in_ptr = input.raw_ptr();
    float* out_ptr = output.raw_ptr();

    #pragma omp parallel for
    for (uint32_t c = 0; c < C; ++c) {
        for (uint32_t i = 0; i < OH; ++i) {
            for (uint32_t j = 0; j < OW; ++j) {
                
                // 寻找局部最大值
                float max_val = -1e9; // 负无穷
                
                // 遍历池化窗口
                for (uint32_t ki = 0; ki < k_size; ++ki) {
                    for (uint32_t kj = 0; kj < k_size; ++kj) {
                        uint32_t h_idx = i * k_size + ki;
                        uint32_t w_idx = j * k_size + kj;
                        
                        // 边界检查 (虽说 stride=k_size 通常不会越界，但为了安全)
                        if (h_idx < H && w_idx < W) {
                             uint32_t offset = c * H * W + h_idx * W + w_idx;
                             float val = in_ptr[offset];
                             if (val > max_val) max_val = val;
                        }
                    }
                }
                
                uint32_t out_offset = c * OH * OW + i * OW + j;
                out_ptr[out_offset] = max_val;
            }
        }
    }
}

void linear(const Tensor& input, const Tensor& weight, const Tensor& bias, Tensor& output) {
    // Linear 其实就是矩阵乘法 Y = X * W^T
    // 我们直接复用 gemm_omp (或者 gemm_avx)
    // 注意：全连接层的权重通常存储为 [Out_Features, In_Features]，需要转置计算
    // 这里为了简单，我们假设输入 weight 已经是 [In, Out] 或者我们调用 GEMM 时注意维度
    
    // 根据申请书设计，Linear 实现为 "向量-矩阵乘法" [cite: 24]
    // 但为了通用性，我们这里视为矩阵乘法
    // input: [Batch, In_Feat]
    // weight: [In_Feat, Out_Feat] (注意这里假设已经转置好了，或者我们在加载模型时转置)
    
    gemm_avx(input, weight, output); // 利用我们最强的算子
    
    // 加上 Bias (偏置)
    // output: [Batch, Out_Feat]
    // bias: [Out_Feat]
    if (bias.size() > 0) {
        uint32_t batch = output.shapes()[0];
        uint32_t out_feat = output.shapes()[1];
        float* out_ptr = output.raw_ptr();
        const float* bias_ptr = bias.raw_ptr();

        #pragma omp parallel for
        for (uint32_t b = 0; b < batch; ++b) {
            for (uint32_t o = 0; o < out_feat; ++o) {
                out_ptr[b * out_feat + o] += bias_ptr[o];
            }
        }
    }
}

} // namespace microflow