#include "tensor.h"
#include "ops.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <string>

using namespace microflow;

// [新增] 全局内存池 (16MB足够容纳 MNIST 所有中间层)
std::vector<float> global_arena(16 * 1024 * 1024);
float* workspace_ptr = global_arena.data();

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "   MicroFlow: MNIST Mock Inference" << std::endl;
    std::cout << "========================================" << std::endl;

    // 1. 打开模型文件
    std::ifstream model_file("mnist.mflow", std::ios::binary);
    if (!model_file.is_open()) {
        std::cerr << "Error: Could not open mnist.mflow! File missing?" << std::endl;
        return -1;
    }

    char magic[4];
    model_file.read(magic, 4);
    if (std::string(magic, 4) != "MFLW") {
        std::cerr << "Error: Invalid file format!" << std::endl;
        return -1;
    }
    std::cout << "[Info] Model file loaded successfully." << std::endl;

    // [1] Input Layer
    Tensor input({1, 28, 28});
    
    // 读取 input.bin
    std::ifstream input_file("input.bin", std::ios::binary);
    if (!input_file.is_open()) {
        std::cerr << "Error: Could not open input.bin!" << std::endl;
        return -1;
    }
    input_file.read(reinterpret_cast<char*>(input.raw_ptr()), 
                    input.size() * sizeof(float));
    input_file.close();
    std::cout << "[Info] Loaded input data from input.bin" << std::endl;

    Tensor conv1_weight({8, 1, 3, 3}); 
    Tensor conv1_out({8, 28, 28});
    
    // 读取 Conv1 权重
    model_file.read(reinterpret_cast<char*>(conv1_weight.raw_ptr()), 
                    conv1_weight.size() * sizeof(float));

    // [关键修正]: 传入 workspace_ptr
    conv2d(input, conv1_weight, conv1_out, 1, 1, workspace_ptr);
    relu(conv1_out);

    Tensor pool1_out({8, 14, 14});
    max_pool2d(conv1_out, pool1_out, 2);

    Tensor fc_weight({1568, 10}); 
    Tensor fc_bias({10});
    Tensor fc_output({1, 10});

    // 读取 FC 权重和 Bias
    model_file.read(reinterpret_cast<char*>(fc_weight.raw_ptr()), fc_weight.size() * sizeof(float));
    model_file.read(reinterpret_cast<char*>(fc_bias.raw_ptr()), fc_bias.size() * sizeof(float));
    model_file.close();

    // Flatten & Linear
    Tensor fc_input_flat({1, 1568});
    // 拷贝 pool1_out 的数据到 fc_input_flat
    std::copy(pool1_out.raw_ptr(), pool1_out.raw_ptr() + pool1_out.size(), fc_input_flat.raw_ptr());
    
    linear(fc_input_flat, fc_weight, fc_bias, fc_output);

    std::cout << "Inference Done! Output Logits (Real Weights):" << std::endl;
    fc_output.show_meta();
    float* out_data = fc_output.raw_ptr();
    for(int i=0; i<10; ++i) {
        std::cout << out_data[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}