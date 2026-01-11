#include "tensor.h"
#include "ops.h"
#include <iostream>
#include <vector>
#include <cmath> // for fabs

bool is_close(float a, float b) {
    return std::fabs(a - b) < 1e-5;
}

int main() {
    using namespace microflow;

    // 1. 准备数据
    // A [2, 3] -> 全是 1
    Tensor A({2, 3});
    for (int i = 0; i < 6; ++i) A.raw_ptr()[i] = 1.0f;

    // B [3, 2] -> 全是 2
    Tensor B({3, 2});
    for (int i = 0; i < 6; ++i) B.raw_ptr()[i] = 2.0f;

    // C [2, 2] -> 结果
    Tensor C({2, 2});

    // 2. 执行矩阵乘法
    // 理论结果：
    // row of A [1, 1, 1] dot col of B [2, 2, 2] = 1*2 + 1*2 + 1*2 = 6
    std::cout << "Running GEMM..." << std::endl;
    gemm_naive(A, B, C);

    // 3. 验证结果
    const float* ptr_C = C.raw_ptr();
    bool pass = true;
    for (int i = 0; i < 4; ++i) {
        if (!is_close(ptr_C[i], 6.0f)) {
            pass = false;
            std::cerr << "Error at index " << i << ": expected 6.0, got " << ptr_C[i] << std::endl;
        }
    }

    if (pass) {
        std::cout << "[PASS] GEMM result correct! All elements are 6.0" << std::endl;
    } else {
        std::cout << "[FAIL] GEMM verification failed." << std::endl;
        return -1;
    }

    return 0;
}