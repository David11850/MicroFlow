# MicroFlow

一个针对 **Raspberry Pi 4 (Cortex-A72)** 进行优化的轻量级推理引擎原型。项目当前包含：

- 基础张量与算子（GEMM / Conv / ReLU / Pool / Linear）
- OpenMP 多线程并行
- x86 AVX2 与 ARM NEON 分支优化
- 面向 Pi4 的简化大模型推理组件（INT8 量化线性层 + KV Cache + Greedy 解码）

## Pi4 大模型推理引擎设计

新增模块：

- `QuantizedMatrix`：按行量化（int8 + per-row scale）
- `KvCache`：缓存历史 token 的 Key/Value，并执行缩放点积注意力
- `Pi4LlmEngine`：
  - 加载 embedding 与量化 lm_head
  - 利用 KV cache 做上下文聚合
  - 使用量化 matvec 计算 logits
  - greedy 解码生成 token

### 适配树莓派4的关键点

1. **内存优先**：权重量化为 int8，显著降低模型驻留内存。
2. **并行优先**：OpenMP 默认可跑满 4 核。
3. **指令集优化**：在 aarch64 下自动启用 NEON 向量路径。
4. **缓存友好**：KV Cache 顺序存储，按时间步线性访问。

## 构建

```bash
cmake -S . -B build
cmake --build build -j
```

## 测试

```bash
./build/test_tensor
./build/test_gemm
./build/test_conv
./build/test_mnist_mock
./build/test_llm_engine
```

## 运行提示（Pi4）

建议在树莓派4上使用 `aarch64` 系统镜像；CMake 会自动追加：

- `-mcpu=cortex-a72`
- `-O3`
- `-ftree-vectorize`

并启用 OpenMP 与 NEON 路径。
