#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <memory>
#include <numeric>
#include <iostream>
#include <algorithm>

namespace microflow {

class Tensor {
public:
    // 使用 shared_ptr 管理数据，引用计数机制防止内存泄漏 
    // float 是推理引擎中最常用的精度
    using Ptr = std::shared_ptr<Tensor>;

    // 构造函数：创建一个全零的 Tensor
    explicit Tensor(const std::vector<uint32_t>& shapes);

    // 核心属性
    const std::vector<uint32_t>& shapes() const { return m_shapes; }
    const std::vector<uint32_t>& strides() const { return m_strides; }
    
    // 获取原始数据指针，用于后续的 SIMD 优化访问
    float* raw_ptr() { return m_data.get(); }
    const float* raw_ptr() const { return m_data.get(); }

    // 获取元素总数
    uint32_t size() const { return m_size; }

    // 打印元数据 (Debug用)
    void show_meta() const;

private:
    std::vector<uint32_t> m_shapes;
    std::vector<uint32_t> m_strides; // 步长，用于从多维坐标计算一维偏移量
    uint32_t m_size;
    
    // 真正存储数据的地方。
    // 在未来的内存优化阶段，我们将把这里的内存分配移交给 MemoryArena [cite: 90]
    std::shared_ptr<float[]> m_data;
};

} // namespace microflow

#endif // TENSOR_H