import torch
import torch.nn as nn

class ResidualBlock1D(nn.Module):
    """标准的一维残差块"""
    def __init__(self, channels):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x  # 记住输入（短路连接）
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual  # 将原始输入加回来
        out = self.relu(out)
        return out

class ResNet1D_Evaluator(nn.Module):
    """用于 FDEM 反演置信度评估的 1D-ResNet"""
    def __init__(self, in_channels=2, hidden_channels=64, num_blocks=4):
        super(ResNet1D_Evaluator, self).__init__()
        
        # 1. 初始卷积特征提取 (将 2 通道升维到 64 通道)
        self.input_layer = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU()
        )
        
        # 2. 堆叠残差块主干
        layers = []
        for _ in range(num_blocks):
            layers.append(ResidualBlock1D(hidden_channels))
        self.res_blocks = nn.Sequential(*layers)
        
        # 3. 输出层 (将 64 通道降维回 1 通道，对应每层的置信度 r)
        self.output_layer = nn.Sequential(
            nn.Conv1d(hidden_channels, 1, kernel_size=3, padding=1),
            # 使用 Sigmoid 强制输出在 (0, 1) 区间，符合置信度的物理意义
            nn.Sigmoid() 
        )

    def forward(self, x):
        # x shape: (Batch, 2, n_layers)
        out = self.input_layer(x)
        out = self.res_blocks(out)
        out = self.output_layer(out)
        # 去掉多余的 channel 维度，变成 (Batch, n_layers)
        return out.squeeze(1)

# 测试代码
if __name__ == "__main__":
    model = ResNet1D_Evaluator()
    dummy_input = torch.randn(16, 2, 20) # 16个样本，2个通道，20层
    output = model(dummy_input)
    print(f"模型输出形状: {output.shape}") # 期望输出: [16, 20]