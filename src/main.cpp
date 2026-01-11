#include <iostream>
#include <fstream>   // 用于文件操作
#include <vector>    // C++11 标准库容器

int main() {
    // 【步骤 1】打开文件
    // 目的：将硬盘上的 mnist.mflow 以二进制流的方式连接到程序中
    // 知识点：std::ios::binary 告诉 C++ 不要把文件当文本读，要原封不动读字节
    std::ifstream file("../model/mnist.mflow", std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "错误：找不到模型文件 mnist.mflow！" << std::endl;
        return -1;
    }

    // 【步骤 2】读取魔数 (Magic Header)
    // 目的：验证文件身份。就像身份证的前四位。
    // 预期结果：应该输出 "MFLW"
    char magic[4]; 
    file.read(magic, 4); 
    std::cout << "1. 文件标识 (Magic): " 
              << magic[0] << magic[1] << magic[2] << magic[3] << std::endl;

    // 【步骤 3】读取版本号
    // 目的：确保 C++ 引擎的版本和导出脚本的版本兼容
    // 待学习点：reinterpret_cast (见下文解析)
    int version = 0;
    file.read(reinterpret_cast<char*>(&version), sizeof(int));
    std::cout << "2. 模型版本: v" << version << std::endl;

    // 【步骤 4】读取模型总层数
    // 目的：让程序知道后面要循环多少次来处理不同的“加工车间”
    int num_layers = 0;
    file.read(reinterpret_cast<char*>(&num_layers), sizeof(int));
    std::cout << "3. 模型总层数: " << num_layers << std::endl;

    // 【步骤 5】关闭文件
    file.close();
    std::cout << "--- 解析完成 ---" << std::endl;

    return 0;
}