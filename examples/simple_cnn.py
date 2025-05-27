"""
简单CNN模型示例
用于测试NetView的基础解析功能
"""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """简单的卷积神经网络"""
    
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # 卷积层
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)
        
        # 全连接层
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        # 激活函数和正则化
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # 卷积+池化层
        x = self.pool(self.relu(self.conv1(x)))  # 32x16x16
        x = self.pool(self.relu(self.conv2(x)))  # 64x8x8
        x = self.pool(self.relu(self.conv3(x)))  # 128x4x4
        
        # 展平
        x = x.view(-1, 128 * 4 * 4)
        
        # 全连接层
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


# 示例用法
if __name__ == "__main__":
    model = SimpleCNN(num_classes=10)
    print(model)
    
    # 测试前向传播
    dummy_input = torch.randn(1, 3, 32, 32)
    output = model(dummy_input)
    print(f"输出形状: {output.shape}")
