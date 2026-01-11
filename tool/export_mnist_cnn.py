import torch
import torch.nn as nn
import torch.nn.functional as F
import struct

# ---------------------------------------------------------
# 1. 定义模型 (必须与 C++ test_mnist_mock.cpp 结构 1:1 对应)
# ---------------------------------------------------------
class MicroCNN(nn.Module):
    def __init__(self):
        super(MicroCNN, self).__init__()
        # Conv2d: In=1, Out=8, Kernel=3, Stride=1, Padding=1
        # C++ 对应: Tensor conv1_weight({8, 1, 3, 3});
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1, bias=False)
        
        # MaxPool 2x2 (无参数，不需要导出)
        
        # Linear: Input=8*14*14 (1568), Output=10
        # C++ 对应: Tensor fc_weight({1568, 10});
        self.fc1 = nn.Linear(8 * 14 * 14, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

def export_weight(filename="mnist.mflow", input_filename="input.bin"):
    print(f"[MicroFlow] 初始化模型与校验数据...")
    torch.manual_seed(1234) # 固定随机种子，保证每次跑结果一样
    model = MicroCNN()
    model.eval()

    # 1. 生成随机输入 (模拟一张图片)
    # Shape: [1, 1, 28, 28]
    x = torch.randn(1, 1, 28, 28)
    
    # 2. PyTorch 推理 (标准答案)
    with torch.no_grad():
        y_gold = model(x)
    
    print(f"\n[PyTorch Output] 标准答案 (Gold Output):")
    print(y_gold.numpy().flatten()) # 打印出来，一会儿跟 C++ 对账

    # 3. 导出模型权重 (mnist.mflow)
    # ... (这部分保持不变，写入 MFLW, Conv1, FC1 ...)
    print(f"\n[MicroFlow] 正在导出模型至 {filename}...")
    with open(filename, "wb") as f:
        f.write(b'MFLW')
        # ... Conv1 Weight ...
        f.write(model.conv1.weight.detach().numpy().tobytes())
        # ... FC1 Weight (Transpose) ...
        f.write(model.fc1.weight.detach().numpy().transpose().tobytes())
        # ... FC1 Bias ...
        f.write(model.fc1.bias.detach().numpy().tobytes())

    # 4. 导出输入数据 (input.bin)
    # 这样 C++ 就可以读入完全相同的 "图片"
    print(f"[MicroFlow] 正在导出输入数据至 {input_filename}...")
    with open(input_filename, "wb") as f:
        f.write(x.numpy().tobytes())

if __name__ == "__main__":
    export_weight()

