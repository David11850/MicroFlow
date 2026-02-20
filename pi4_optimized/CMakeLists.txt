cmake_minimum_required(VERSION 3.16)
project(MicroFlow
    VERSION 2.0
    DESCRIPTION "Lightweight Inference Engine for Raspberry Pi 4"
    LANGUAGES CXX
)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 构建类型
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ============================================================================
# 树莓派4 (ARM64) 优化选项
# ============================================================================
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64")
    message(STATUS "Configuring for Raspberry Pi 4 (ARM64)")

    # Cortex-A72优化
    add_compile_options(
        -march=armv8-a
        -mtune=cortex-a72
        -mcpu=cortex-a72
    )

    # 启用NEON和FP16
    add_compile_options(
        -mfpu=neon-fp-armv8
        -mfloat-abi=hard
    )

    #激进优化
    add_compile_options(-O3)

    #数学优化
    add_compile_options(
        -ffast-math
        -funsafe-math-optimizations
        -ffp-contract=fast
    )

    # 循环优化
    add_compile_options(
        -funroll-loops
        -ftree-vectorize
    )

    # 链接时优化
    add_compile_options(-flto)
    add_link_options(-flto)

    # 硬标注: 树莓派4有4个核心
    set(DEFAULT_NUM_THREADS 4)

elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    message(STATUS "Configuring for x86_64 (development)")

    add_compile_options(-O3)
    add_compile_options(
        -march=native
        -ffast-math
        -funroll-loops
        -ftree-vectorize
    )

    set(DEFAULT_NUM_THREADS 4)

else()
    message(STATUS "Configuring for generic architecture")
    add_compile_options(-O3)
    set(DEFAULT_NUM_THREADS 2)
endif()

# ============================================================================
# 编译器警告
# ============================================================================
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
        -Wall
        -Wextra
        -Wpedantic
        -Wno-unused-parameter
    )
endif()

# ============================================================================
# 依赖项
# ============================================================================

# OpenMP (多线程支持)
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
    target_compile_options(microflow_core PUBLIC OpenMP::OpenMP_CXX)
else()
    message(FATAL_ERROR "OpenMP is required but not found")
endif()

# ============================================================================
# 包含目录
# ============================================================================
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/microflow
)

# ============================================================================
# 微流核心库
# ============================================================================
add_library(microflow_core STATIC
    # 内存管理
    src/memory/allocator.cpp
    src/memory/tensor.cpp

    # GEMM
    src/gemm/gemm.cpp

    # 卷积
    src/conv/conv.cpp

    # 层操作
    src/layers/layers.cpp

    # 运行时
    src/runtime/runtime.cpp
)

# 链接OpenMP
target_link_libraries(microflow_core
    PUBLIC
        OpenMP::OpenMP_CXX
)

# 设置库版本
set_target_properties(microflow_core PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 2
    PUBLIC_HEADER "include/microflow/*.hpp"
)

# ============================================================================
# 测试程序
# ============================================================================

# 内存分配器测试
add_executable(test_allocator
    tests/test_allocator.cpp
)
target_link_libraries(test_allocator microflow_core)

# Tensor测试
add_executable(test_tensor
    tests/test_tensor.cpp
)
target_link_libraries(test_tensor microflow_core)

# GEMM测试
add_executable(test_gemm
    tests/test_gemm.cpp
)
target_link_libraries(test_gemm microflow_core)

# 卷积测试
add_executable(test_conv
    tests/test_conv.cpp
)
target_link_libraries(test_conv microflow_core)

# 性能基准测试
add_executable(benchmark
    tests/benchmark.cpp
)
target_link_libraries(benchmark microflow_core)

# ============================================================================
# 示例程序
# ============================================================================

# MNIST推理示例
add_executable(mnist_demo
    examples/mnist_demo.cpp
)
target_link_libraries(mnist_demo microflow_core)

# ============================================================================
# 安装
# ============================================================================
install(TARGETS microflow_core
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    PUBLIC_HEADER DESTINATION include/microflow
)

install(FILES
    README.md
    LICENSE
    DESTINATION .
)

# ============================================================================
# CPack (打包)
# ============================================================================
set(CPACK_PACKAGE_NAME "microflow")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Lightweight Inference Engine")
set(CPACK_PACKAGE_CONTACT "your-email@example.com")

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "arm64")
    set(CPACK_RPM_PACKAGE_ARCHITECTURE "aarch64")
endif()

include(CPack)
