cmake_minimum_required(VERSION 3.10)
project(MicroFlow)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 开启原生架构优化,针对树莓派 4 (Cortex-A72) 进行精准优化
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    # -O3: 激进优化
    # -mcpu=cortex-a72: 告诉编译器这是树莓派 4 的心脏
    # -ftree-vectorize: 强制开启自动向量化（利用 NEON 指令集）
    add_compile_options(-O3 -mcpu=cortex-a72 -ftree-vectorize)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    add_compile_options(-O3 -mavx2 -mfma)
else()
    add_compile_options(-O3)
endif()

# 寻找 OpenMP (Level 1 优化必须)
find_package(OpenMP REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)

# 1. 编译核心库
add_library(microflow_core src/tensor.cpp src/ops.cpp src/im2col.cpp src/llm_engine.cpp)
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
add_executable(test_llm_engine tests/test_llm_engine.cpp)
target_link_libraries(test_llm_engine microflow_core)
