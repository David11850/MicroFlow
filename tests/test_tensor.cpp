#include "tensor.h"
#include <iostream>

int main() {
    using namespace microflow;
    
    // 创建一个 [2, 3, 4] 的 3D 张量
    std::vector<uint32_t> shape = {2, 3, 4};
    Tensor tensor(shape);
    
    tensor.show_meta();
    
    if (tensor.size() == 2 * 3 * 4) {
        std::cout << "\n[PASS] Tensor initialization successful!" << std::endl;
    } else {
        std::cerr << "\n[FAIL] Tensor size mismatch!" << std::endl;
        return -1;
    }
    
    return 0;
}