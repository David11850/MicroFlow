# 阶段 1：编译环境 (Build Stage)
FROM docker.m.daocloud.io/library/debian:13-slim AS builder

# 安装编译所需的依赖
RUN apt update && apt install -y \
    build-essential \
    cmake \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 拷贝源代码到镜像中
COPY . .

# 执行编译（针对 ARM64 的本地编译优化）
RUN mkdir -p build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc)

# 阶段 2：运行环境 (Runtime Stage)
FROM docker.m.daocloud.io/library/debian:13-slim

# 只安装运行所需的运行时库（OpenMP）
RUN apt update && apt install -y \
    libomp5 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /root/microflow/

# 从编译阶段拷贝生成的可执行文件和模型文件
COPY --from=builder /app/build/test_mnist_mock .
COPY --from=builder /app/model/mnist.mflow ./model/

# 设置启动指令
ENTRYPOINT ["./test_mnist_mock"]