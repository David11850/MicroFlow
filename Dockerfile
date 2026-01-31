# ==========================================
# 阶段 1：编译阶段 (Builder) - 负责交叉编译
# ==========================================
FROM docker.m.daocloud.io/library/debian:13-slim AS builder

# 安装编译所需的完整工具链(g++/cmake/openmp)
RUN apt update && apt install -y build-essential cmake libgomp1

WORKDIR /app
# 将当前目录所有文件拷贝进镜像
COPY . .

# 内部执行 CMake 和 Make（生成 ARM64 二进制）
RUN mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make test_mnist_mock




# ==========================================
# 阶段 2：运行阶段 (Runtime) - 负责极简部署
# ==========================================
FROM docker.m.daocloud.io/library/debian:13-slim

# 只安装运行时必需的库
RUN apt update && apt install -y libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /root/microflow

# 1. 从编译阶段拷贝最终的可执行文件
COPY --from=builder /app/build/test_mnist_mock .

# 2. 固化模型文件到镜像内部
COPY --from=builder /app/model/ ./model/

# 3. 固化你生成的那个数字 7 的测试图片到镜像内部
COPY --from=builder /app/tool/input.bin .

# 赋予程序执行权限
RUN chmod +x ./test_mnist_mock

# 一键启动：不带任何参数默认运行推理
ENTRYPOINT ["./test_mnist_mock"]