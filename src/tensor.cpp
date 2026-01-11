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

// ---------------------------------------------------------
// [新增实现] Zero-Copy 构造函数
// ---------------------------------------------------------
Tensor::Tensor(const std::vector<uint32_t>& shapes, float* external_ptr)
    : m_shapes(shapes) 
{
    // 1. 计算总元素大小和步长 (复用之前的逻辑)
    m_size = 1;
    m_strides.resize(shapes.size());
    
    if (!shapes.empty()) {
        // 计算步长 (Row-Major)
        std::vector<uint32_t> temp_strides(shapes.size());
        uint32_t running_stride = 1;
        for (int i = shapes.size() - 1; i >= 0; --i) {
            temp_strides[i] = running_stride;
            running_stride *= shapes[i];
        }
        m_strides = temp_strides;
        m_size = running_stride;
    }

    // 2. 关键点：使用自定义删除器
    // "[](float* p) {}" 是一个空函数，意味着：
    // 当 m_data 引用计数归零时，不要 delete p！因为这块内存是外面传进来的 workspace。
    m_data = std::shared_ptr<float[]>(external_ptr, [](float* p) { 
        // Do Nothing! 
    });
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