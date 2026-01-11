#include "tensor.h"
#include <iomanip>

namespace microflow {

Tensor::Tensor(const std::vector<uint32_t>& shapes) 
    : m_shapes(shapes) {
    
    if (shapes.empty()) {
        m_size = 0;
        return;
    }

    // 1. 计算总元素数量
    m_size = std::accumulate(shapes.begin(), shapes.end(), 1, std::multiplies<uint32_t>());

    // 2. 分配内存 (目前直接使用 new，后续我们将引入 Memory Pool [cite: 29])
    // 这里的 float() 会将内存初始化为 0
    m_data = std::shared_ptr<float[]>(new float[m_size](), std::default_delete<float[]>());

    // 3. 计算 Strides (行主序)
    // 例如 shape [2, 3, 4] -> strides [12, 4, 1]
    m_strides.resize(shapes.size());
    uint32_t stride = 1;
    for (int i = shapes.size() - 1; i >= 0; --i) {
        m_strides[i] = stride;
        stride *= shapes[i];
    }
}

void Tensor::show_meta() const {
    std::cout << "Tensor Meta Info:\n";
    std::cout << "  Shape: [";
    for (auto s : m_shapes) std::cout << s << " ";
    std::cout << "]\n";
    
    std::cout << "  Strides: [";
    for (auto s : m_strides) std::cout << s << " ";
    std::cout << "]\n";
    
    std::cout << "  Total Elements: " << m_size << "\n";
    std::cout << "  Memory Addr: " << m_data.get() << "\n";
}

} // namespace microflow