cmake_minimum_required(VERSION 3.10)
project(MicroFlow)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 开启原生架构优化 (自动识别你的 CPU 支持 AVX2/FMA 等指令)
# 注意：在之后的交叉编译环节，这行需要删掉或改为 ARM 专用参数
add_compile_options(-O3 -march=native)

# 寻找 OpenMP (Level 1 优化必须)
find_package(OpenMP REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)

# 1. 编译核心库
add_library(microflow_core src/tensor.cpp src/ops.cpp src/im2col.cpp)
target_link_libraries(microflow_core PUBLIC OpenMP::OpenMP_CXX)

# 2. 之前的测试
add_executable(test_tensor tests/test_tensor.cpp)
target_link_libraries(test_tensor microflow_core)

add_executable(test_gemm tests/test_gemm.cpp)
target_link_libraries(test_gemm microflow_core)

add_executable(benchmark_gemm tests/benchmark_gemm.cpp)
target_link_libraries(benchmark_gemm microflow_core)

# 3. 新增卷积测试
add_executable(test_conv tests/test_conv.cpp)
target_link_libraries(test_conv microflow_core)

add_executable(test_mnist_mock tests/test_mnist_mock.cpp)
target_link_libraries(test_mnist_mock microflow_core)