import numpy as np

def verify_bin(file_path):
    # 1. 以 float32 格式读取数据
    data = np.fromfile(file_path, dtype=np.float32)
    
    # 2. 检查数据长度是否为 784 (28*28)
    if data.size != 784:
        print(f"数据长度错误: {data.size}")
        return

    # 3. 重塑并打印 ASCII 字符画
    # 我们用像素值来决定显示 '#' 还是 '.'
    img = data.reshape(28, 28)
    print("\n=== standard_input.bin 可视化内容 ===")
    for row in img:
        # 只要像素值大于 0.2 就打印 #，否则打印 .
        line = "".join(["#" if val > 0.2 else "." for val in row])
        print(line)
    print("======================================\n")

if __name__ == "__main__":
    verify_bin("standard_input.bin")