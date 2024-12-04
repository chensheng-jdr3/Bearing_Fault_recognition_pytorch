import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        
        # 下采样
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, 1, stride)
        else:
            self.downsample = nn.Identity()
    
    def forward(self, x):
        identity = self.downsample(x)
        
        # 第一个卷积单元
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        # 第二个卷积单元
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        
        # 残差连接
        out = out + identity
        out = F.relu(out)
        
        return out


# ResNet定义类
class ResNet(nn.Module):
    def __init__(self, layer_dims, num_classes=4):
        super(ResNet, self).__init__()
        
        self.stem = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        
        self.layer1 = self._make_layer(16, layer_dims[0], in_channels=16)
        self.layer2 = self._make_layer(32, layer_dims[1], kernel_size=5, stride=4)
        self.layer3 = self._make_layer(64, layer_dims[2], kernel_size=5, stride=4)
        self.layer4 = self._make_layer(128, layer_dims[3], stride=2)
        self.layer5 = self._make_layer(256, layer_dims[4], stride=2)
        self.layer6 = self._make_layer(512, layer_dims[5], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)  # 自适应池化，输出大小为1
        self.fc = nn.Linear(512, num_classes)  # 输出num_classes个类别
    
    def _make_layer(self, out_channels, blocks, kernel_size=3, stride=1, in_channels=None):
        if in_channels is None:
            in_channels = out_channels//2  # 如果未指定，则假设输入通道数等于输出通道数
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, kernel_size, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, kernel_size))
        return nn.Sequential(*layers)

    
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layer
        x = self.fc(x)
        return x


# 创建 ResNet-18 实例
def create_ResNet():
    return ResNet([2, 2, 2, 2, 2, 2], num_classes=4)

# 示例使用
if __name__ == "__main__":
    # 创建 ResNet 网络
    model = create_ResNet()

    # 输出网络架构
    print(model)
    