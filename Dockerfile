# ==========================================
# 阶段 1：编译环境 (Builder Stage)
# ==========================================
FROM docker.m.daocloud.io/library/debian:13-slim AS builder

# 安装编译所需的完整工具链（针对 GCC 编译）
RUN apt update && apt install -y \
    build-essential \
    cmake \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 只有在构建时才进行拷贝，避免宿主机 build 文件夹污染
COPY . .

# 执行编译（在 ARM64 模拟环境下生成原生二进制）
# 强制清理容器内可能存在的旧 build
RUN rm -rf build && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc)

# ==========================================
# 阶段 2：运行环境 (Runtime Stage)
# ==========================================
FROM docker.m.daocloud.io/library/debian:13-slim

# 设置工作目录
WORKDIR /root/microflow/

# 核心修正：安装与编译器匹配的 GNU OpenMP 运行时库
RUN apt update && apt install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 从编译阶段拷贝成品，不要拷贝整个项目，保持镜像轻量
COPY --from=builder /app/build/test_mnist_mock .
COPY --from=builder /app/model/ ./model/

# 赋予执行权限
RUN chmod +x ./test_mnist_mock

# 设置启动命令
ENTRYPOINT ["./test_mnist_mock"]