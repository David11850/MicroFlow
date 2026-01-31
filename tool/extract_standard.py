import numpy as np
import os

def extract_one_mnist(file_path, output_path, index=0):
    # 以二进制模式读取原始 ubyte 文件
    with open(file_path, 'rb') as f:
        # 跳过 16 字节的文件头 (Magic number, num_images, rows, cols)
        f.seek(16 + index * 28 * 28)
        # 读取一张图的数据 (784 字节)
        raw_data = f.read(28 * 28)
        
        # 将 uint8 (0-255) 转换为 float32 (0.0-1.0)
        # 这一步非常关键，必须匹配你 C++ 推理引擎的 float 格式
        pixels = np.frombuffer(raw_data, dtype=np.uint8).astype(np.float32) / 255.0
        
        # 导出为 bin 文件
        pixels.tofile(output_path)
        print(f"成功提取第 {index} 张图并保存至: {output_path}")

if __name__ == "__main__":
    # 根据你的 tree 结构定位原始文件
    raw_file = "data/MNIST/raw/t10k-images-idx3-ubyte"
    if os.path.exists(raw_file):
        extract_one_mnist(raw_file, "standard_input.bin", index=0) # 提取第一张图
    else:
        print("未发现原始文件，请确保路径正确或已运行 train_and_export.py 下载数据")