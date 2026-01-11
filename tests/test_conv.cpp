#include "tensor.h"
#include "ops.h" // 包含 conv2d
#include <iostream>
#include <cmath> // for fabs
#include <vector>

// 浮点数比较辅助函数
bool is_close(float a, float b) {
    return std::fabs(a - b) < 1e-5;
}

int main() {
    using namespace microflow;
    
    std::cout << "Test Case: Conv2d 3x3 Input, 2x2 Kernel" << std::endl;

    // 1. 构造输入 (Input Image)
    // Shape: [Channels=1, Height=3, Width=3]
    Tensor input({1, 3, 3});
    // "造"图片：将所有像素设为 1.0
    float* in_ptr = input.raw_ptr();
    for(int i=0; i < input.size(); ++i) {
        in_ptr[i] = 1.0f; 
    }

    // 2. 构造卷积核 (Kernel)
    // Shape: [Fn=1, C=1, K=2, K=2]
    // 1个卷积核，对应1个输入通道，尺寸2x2
    Tensor kernel({1, 1, 2, 2});
    float* k_ptr = kernel.raw_ptr();
    for(int i=0; i < kernel.size(); ++i) {
        k_ptr[i] = 1.0f;
    }

    // 3. 准备输出 (Output)
    // 根据公式算出结果应该是 [1, 2, 2]
    Tensor output({1, 2, 2});

    // 4. 执行卷积
    // Stride = 1, Padding = 0
    std::cout << "Running Conv2d..." << std::endl;
    conv2d(input, kernel, output, 1, 0);

    // 5. 验证结果
    const float* out_ptr = output.raw_ptr();
    bool pass = true;
    
    std::cout << "Output Matrix:" << std::endl;
    for (int i = 0; i < 2; ++i) { // H
        for (int j = 0; j < 2; ++j) { // W
             int idx = i * 2 + j;
             float val = out_ptr[idx];
             std::cout << val << " ";
             
             // 核心验证：必须等于 4.0
             if (!is_close(val, 4.0f)) {
                 pass = false;
             }
        }
        std::cout << std::endl;
    }

    if (pass) {
        std::cout << "\n[PASS] Conv2d result correct! All elements are 4.0" << std::endl;
    } else {
        std::cerr << "\n[FAIL] Conv2d result incorrect!" << std::endl;
        return -1;
    }

    return 0;
}