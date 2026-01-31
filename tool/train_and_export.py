import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import struct
import os

# 1. 定义和 C++端完全一致的模型结构
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Conv1: input 1x28x28 -> output 8x26x26 (kernel=3, no padding)
        # 注意：你的C++代码里是 input 28x28, kernel 3x3, stride 1
        # 如果没有 padding，输出应该是 26x26。
        # 但你的 C++ 里 input 是 28x28，输出依然写的是 28x28？
        # 让我们先按照你的 C++ 代码逻辑来凑权重。
        
        # 修正：为了配合你的 C++ 里的 "Conv1 weights: 8, 1, 3, 3"
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1) 
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 经过 28x28 -> (padding=1) -> 28x28 -> (pool=2) -> 14x14
        # Flatten: 8 * 14 * 14 = 1568
        self.fc = nn.Linear(1568, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(-1, 1568)
        x = self.fc(x)
        return x

def train_and_export():
    # 2. 准备数据
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    # 3. 初始化并训练
    print("正在训练简易模型 (约需 1 分钟)...")
    model = SimpleCNN()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx > 100: break # 为了演示速度，只训练 100 个 batch，足够识别数字 7 了

    print("训练完成！开始导出 mnist.mflow...")

    # 4. 导出权重 (和你之前的逻辑一致)
    with open("../model/mnist.mflow", "wb") as f:
        f.write(b"MFLW") # Magic Number
        
        # 导出 Conv1 权重 (8, 1, 3, 3)
        # PyTorch 格式: [out_channels, in_channels, kH, kW] -> 直接写入即可
        # 注意 C++ 读取顺序，如果 C++ 是按行优先，PyTorch tensor 也是行优先，通常直接 dump 没问题
        f.write(model.conv1.weight.data.numpy().tobytes())
        # 注意：你的 C++ 代码里好像没读 conv1 的 bias？如果没有，这里就不存
        
        # 导出 FC 权重 (10, 1568)
        # PyTorch Linear 权重是 [out_features, in_features]
        # 你的 C++ Tensor fc_weight({1568, 10}) 看起来是转置过的？
        # 检查你的 C++ GEMM 实现：
        # 如果是 input * weight，那么 weight 应该是 [1568, 10]
        # PyTorch 是 y = xA^T + b，所以 PyTorch 的 weight 是 [10, 1568]
        # **关键点**：这里需要转置一下才能匹配你的 C++ 乘法习惯
        fc_w = model.fc.weight.data.numpy().transpose() # [1568, 10]
        f.write(fc_w.tobytes())
        
        # 导出 FC Bias
        f.write(model.fc.bias.data.numpy().tobytes())

    print("导出成功！请重新构建 Docker 镜像。")

if __name__ == "__main__":
    train_and_export()