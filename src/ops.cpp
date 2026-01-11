#include "ops.h"
#include <cassert>
#include <omp.h>
#include "im2col.h"
#include <algorithm> // for std::min

// ---------------- HARDWARE ABSTRACTION LAYER ----------------
#if defined(__x86_64__) || defined(_M_X64)
    #include <immintrin.h>
    #define ENABLE_AVX2 
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
    #include <arm_neon.h>
    #define ENABLE_NEON 
#endif
// ------------------------------------------------------------

namespace microflow {

// ==========================================
//           Common Implementations
// ==========================================

void gemm_naive(const Tensor& A, const Tensor& B, Tensor& C) {
    // ... (保持之前的实现，为了节省篇幅我略去，请保留你的原始内容) ...
    // 如果你之前的代码这里有内容，请不要删掉，保持原样即可
    uint32_t M = A.shapes()[0];
    uint32_t K = A.shapes()[1];
    uint32_t N = B.shapes()[1];
    const float* ptr_A = A.raw_ptr();
    const float* ptr_B = B.raw_ptr();
    float* ptr_C = C.raw_ptr();
    for (uint32_t i = 0; i < M; ++i) {            
        for (uint32_t j = 0; j < N; ++j) {        
            float sum = 0.0f;
            for (uint32_t k = 0; k < K; ++k) {    
                sum += ptr_A[i * A.strides()[0] + k * A.strides()[1]] * ptr_B[k * B.strides()[0] + j * B.strides()[1]];
            }
            ptr_C[i * C.strides()[0] + j * C.strides()[1]] = sum;
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
                sum += ptr_A[i * K + k] * ptr_B[k * N + j];
            }
            ptr_C[i * N + j] = sum;
        }
    }
}

// ==========================================
//           x86 AVX2 Implementation
// ==========================================
#ifdef ENABLE_AVX2 

#define BLOCK_SIZE 64

void gemm_micro_kernel_avx(int M, int N, int K, 
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
                gemm_micro_kernel_avx(M_cur, N_cur, K_cur, 
                                      ptr_A + i * K + k, K, 
                                      ptr_B + k * N + j, N, 
                                      ptr_C + i * N + j, N);
            }
        }
    }
}
#else 
void gemm_avx(const Tensor& A, const Tensor& B, Tensor& C) {
    std::cerr << "[Error] AVX2 not supported!" << std::endl; exit(-1);
}
#endif 

// ==========================================
//           ARM NEON Implementation (NEW!)
// ==========================================
#ifdef ENABLE_NEON

#define BLOCK_SIZE_NEON 64

// [核心] NEON 微内核：一次处理 4 个 float
void gemm_micro_kernel_neon(int M, int N, int K, 
                            const float* A, int lda, 
                            const float* B, int ldb, 
                            float* C, int ldc) {
    for (int i = 0; i < M; ++i) {
        int j = 0;
        // 1. SIMD Loop: 每次步进 4
        for (; j <= N - 4; j += 4) { 
            // Load C (Accumulator)
            float32x4_t sum_vec = vld1q_f32(&C[i * ldc + j]); 
            
            for (int k = 0; k < K; ++k) {
                // Broadcast A: 将 A[i][k] 复制 4 份填满向量
                float32x4_t vec_A = vdupq_n_f32(A[i * lda + k]);
                
                // Load B: 读取 B[k][j...j+3]
                float32x4_t vec_B = vld1q_f32(&B[k * ldb + j]);
                
                // FMA: sum = sum + A * B
                // vmlaq_f32 是 ARMv8 的乘加指令 (Vector Multiply Accumulate)
                sum_vec = vmlaq_f32(sum_vec, vec_A, vec_B);
            }
            // Store C
            vst1q_f32(&C[i * ldc + j], sum_vec);
        }
        
        // 2. Remainder Loop: 处理剩下的尾巴 (比如 N=10，剩下 2 个)
        for (; j < N; ++j) {
            float sum = C[i * ldc + j];
            for (int k = 0; k < K; ++k) {
                sum += A[i * lda + k] * B[k * ldb + j];
            }
            C[i * ldc + j] = sum;
        }
    }
}

void gemm_neon(const Tensor& A, const Tensor& B, Tensor& C) {
    uint32_t M = A.shapes()[0];
    uint32_t K = A.shapes()[1];
    uint32_t N = B.shapes()[1];
    const float* ptr_A = A.raw_ptr();
    const float* ptr_B = B.raw_ptr();
    float* ptr_C = C.raw_ptr();

    // 依然使用 OpenMP 做多线程分块，内部使用 NEON
    #pragma omp parallel for
    for (uint32_t i = 0; i < M; i += BLOCK_SIZE_NEON) {
        for (uint32_t k = 0; k < K; k += BLOCK_SIZE_NEON) {
            for (uint32_t j = 0; j < N; j += BLOCK_SIZE_NEON) {
                int M_cur = std::min((uint32_t)BLOCK_SIZE_NEON, M - i);
                int K_cur = std::min((uint32_t)BLOCK_SIZE_NEON, K - k);
                int N_cur = std::min((uint32_t)BLOCK_SIZE_NEON, N - j);
                
                gemm_micro_kernel_neon(M_cur, N_cur, K_cur, 
                                       ptr_A + i * K + k, K, 
                                       ptr_B + k * N + j, N, 
                                       ptr_C + i * N + j, N);
            }
        }
    }
}
#else
void gemm_neon(const Tensor& A, const Tensor& B, Tensor& C) {
    std::cerr << "[Warning] NEON not supported." << std::endl;
}
#endif 

// ==========================================
//           Kernel Dispatcher (智能调度)
// ==========================================
// 这个函数是关键：它决定了在不同硬件上用谁
void gemm_best(const Tensor& A, const Tensor& B, Tensor& C) {
#if defined(ENABLE_AVX2)
    gemm_avx(A, B, C); // x86 电脑用这个
#elif defined(ENABLE_NEON)
    gemm_neon(A, B, C); // RDK S100 板子用这个
#else
    gemm_omp(A, B, C); // 都没有就用 OpenMP
#endif
}

// ==========================================
//           High-Level Ops
// ==========================================

void conv2d(const Tensor& input, const Tensor& kernel, Tensor& output, 
            uint32_t stride, uint32_t padding, float* workspace) {
    // ... (前置逻辑不变) ...
    uint32_t H = input.shapes()[1]; 
    uint32_t W = input.shapes()[2];
    uint32_t FN = kernel.shapes()[0];
    uint32_t C  = kernel.shapes()[1];
    uint32_t K  = kernel.shapes()[2];
    uint32_t OH = (H + 2 * padding - K) / stride + 1;
    uint32_t OW = (W + 2 * padding - K) / stride + 1;

    if (workspace == nullptr) { std::exit(-1); }

    Tensor col_buffer({C * K * K, OH * OW}, workspace); 
    im2col(input, col_buffer, K, stride, padding);

    Tensor kernel_mat({FN, C*K*K});
    std::copy(kernel.raw_ptr(), kernel.raw_ptr() + kernel.size(), kernel_mat.raw_ptr());
    Tensor output_mat({FN, OH * OW});

    // [关键修改]: 使用智能调度，而不是死板的 gemm_omp
    gemm_best(kernel_mat, col_buffer, output_mat);

    std::copy(output_mat.raw_ptr(), output_mat.raw_ptr() + output_mat.size(), output.raw_ptr());
}

void linear(const Tensor& input, const Tensor& weight, const Tensor& bias, Tensor& output) {
    // [关键修改]: 也是用 gemm_best
    gemm_best(input, weight, output);
    
    // Bias 处理保持不变
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

// ... relu, max_pool2d 保持不变 ...
void relu(Tensor& tensor) {
    uint32_t size = tensor.size();
    float* ptr = tensor.raw_ptr();
    #pragma omp parallel for
    for (uint32_t i = 0; i < size; ++i) { ptr[i] = std::max(0.0f, ptr[i]); }
}

void max_pool2d(const Tensor& input, Tensor& output, uint32_t k_size) {
    // ... (保持你之前的 max_pool2d 代码，不要动) ...
     uint32_t C = input.shapes()[0];
    uint32_t H = input.shapes()[1];
    uint32_t W = input.shapes()[2];
    uint32_t OH = output.shapes()[1];
    uint32_t OW = output.shapes()[2];
    const float* in_ptr = input.raw_ptr();
    float* out_ptr = output.raw_ptr();
    #pragma omp parallel for
    for (uint32_t c = 0; c < C; ++c) {
        for (uint32_t i = 0; i < OH; ++i) {
            for (uint32_t j = 0; j < OW; ++j) {
                float max_val = -1e9;
                for (uint32_t ki = 0; ki < k_size; ++ki) {
                    for (uint32_t kj = 0; kj < k_size; ++kj) {
                        uint32_t h_idx = i * k_size + ki;
                        uint32_t w_idx = j * k_size + kj;
                        if (h_idx < H && w_idx < W) {
                             float val = in_ptr[c * H * W + h_idx * W + w_idx];
                             if (val > max_val) max_val = val;
                        }
                    }
                }
                out_ptr[c * OH * OW + i * OW + j] = max_val;
            }
        }
    }
}

} // namespace microflow