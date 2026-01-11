#include "tensor.h"
#include "ops.h"
#include <iostream>
#include <chrono> // 用于高精度计时
#include <vector>
#include <iomanip> // 用于格式化输出

using namespace microflow;

// 计时辅助函数：接收一个 lambda 函数并执行，返回毫秒数
template<typename Func>
long long measure_ms(Func func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

int main() {
    // ---------------------------------------------------------
    // 1. 准备数据
    // ---------------------------------------------------------
    // 为了让 AVX2 跑得爽，我们用 512x512，正好是 8 的倍数
    uint32_t M = 1024, K = 1024, N = 1024;
    
    std::cout << "========================================" << std::endl;
    std::cout << "MicroFlow Benchmark: GEMM [" << M << "x" << N << "]" << std::endl;
    std::cout << "========================================" << std::endl;

    Tensor A({M, K});
    Tensor B({K, N});
    
    // 准备三个结果容器，分别存放三种算法的结果
    Tensor C_naive({M, N});
    Tensor C_omp({M, N});
    Tensor C_avx({M, N});

    // 填充数据 (全为 1.0f，简化验证)
    std::fill(A.raw_ptr(), A.raw_ptr() + A.size(), 1.0f);
    std::fill(B.raw_ptr(), B.raw_ptr() + B.size(), 1.0f);

    // ---------------------------------------------------------
    // 2. 基准测试：Naive (Level 0)
    // ---------------------------------------------------------
    std::cout << "\n1. Running Naive Implementation (Baseline)..." << std::endl;
    long long t_naive = measure_ms([&]() {
        gemm_naive(A, B, C_naive);
    });
    std::cout << "   -> Time: " << t_naive << " ms" << std::endl;

    // ---------------------------------------------------------
    // 3. 基准测试：OpenMP (Level 1)
    // ---------------------------------------------------------
    std::cout << "\n2. Running OpenMP Implementation (Multi-threading)..." << std::endl;
    long long t_omp = measure_ms([&]() {
        gemm_omp(A, B, C_omp);
    });
    std::cout << "   -> Time: " << t_omp << " ms" << std::endl;

    // ---------------------------------------------------------
    // 4. 基准测试：AVX2 + OpenMP (Level 2)
    // ---------------------------------------------------------
    std::fill(C_avx.raw_ptr(), C_avx.raw_ptr() + C_avx.size(), 0.0f);
    std::cout << "\n3. Running AVX2 SIMD Implementation (Blocked)..." << std::endl;    long long t_avx = measure_ms([&]() {
        gemm_avx(A, B, C_avx);
    });
    std::cout << "   -> Time: " << t_avx << " ms" << std::endl;

    // ---------------------------------------------------------
    // 5. 结果汇总与对比
    // ---------------------------------------------------------
    std::cout << "\n========================================" << std::endl;
    std::cout << "Performance Summary" << std::endl;
    std::cout << "========================================" << std::endl;
    
    double speedup_omp = (double)t_naive / t_omp;
    double speedup_avx = (double)t_naive / t_avx;

    std::cout << std::left << std::setw(15) << "Method" 
              << std::setw(15) << "Time (ms)" 
              << "Speedup" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    
    std::cout << std::left << std::setw(15) << "Naive" 
              << std::setw(15) << t_naive 
              << "1.0x (Baseline)" << std::endl;
              
    std::cout << std::left << std::setw(15) << "OpenMP" 
              << std::setw(15) << t_omp 
              << speedup_omp << "x" << std::endl;
              
    std::cout << std::left << std::setw(15) << "AVX2+OMP" 
              << std::setw(15) << t_avx 
              << speedup_avx << "x" << std::endl;

    // ---------------------------------------------------------
    // 6. 简单正确性检查 (抽查)
    // ---------------------------------------------------------
    // 修正：理论值应该是 K (1024.0)，而不是 512.0
    float expected_val = (float)K; 
    
    if (std::abs(C_avx.raw_ptr()[0] - expected_val) < 1e-4) {
        std::cout << "\n[Status] AVX2 Calculation Correct." << std::endl;
    } else {
        std::cerr << "\n[WARNING] AVX2 Calculation WRONG! Expected " << expected_val 
                  << ", got " << C_avx.raw_ptr()[0] << std::endl;
    }

    return 0;
}