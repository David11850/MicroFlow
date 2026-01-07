import torch
import torch.nn as nn
import struct

# 定义简单的模型结构 (对应设计文档 2.3 章节) [cite: 36, 41]
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 128) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def export_mflow(model, filename="mnist.mflow"):
    print(f"[MicroFlow] 正在导出模型至 {filename}...")
    with open(filename, "wb") as f:
        # 1. Magic Header: "MFLW" + Version 1 
        f.write(b'MFLW')
        f.write(struct.pack('i', 1))
        
        # 2. Global Meta: 层数 (3层: FC1, ReLU, FC2) 
        f.write(struct.pack('i', 3))
        
        # 3. 导出第一层 (FC1) [cite: 48, 59]
        f.write(struct.pack('i', 1)) # ID 1 代表 Linear
        w1 = model.fc1.weight.detach().numpy()
        b1 = model.fc1.bias.detach().numpy()
        f.write(struct.pack('iii', 784, 128, (w1.size + b1.size) * 4)) # 输入, 输出, 字节长度
        f.write(w1.tobytes())
        f.write(b1.tobytes())
        
        # 4. 导出第二层 (ReLU) [cite: 47, 59]
        f.write(struct.pack('i', 2)) # ID 2 代表 ReLU
        f.write(struct.pack('iii', 128, 128, 0)) # 无权重数据
        
        # 5. 导出第三层 (FC2) [cite: 48, 59]
        f.write(struct.pack('i', 1)) # ID 1 代表 Linear
        w2 = model.fc2.weight.detach().numpy()
        b2 = model.fc2.bias.detach().numpy()
        f.write(struct.pack('iii', 128, 10, (w2.size + b2.size) * 4))
        f.write(w2.tobytes())
        f.write(b2.tobytes())

if __name__ == "__main__":
    net = SimpleNet()
    # 这里我们跳过漫长的训练，直接用随机初始化的权重导出，先跑通 C++ 流程
    export_mflow(net)
    print("[MicroFlow] 导出成功！")
