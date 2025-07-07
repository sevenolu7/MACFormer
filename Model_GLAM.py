import torch
import torch.nn as nn
import torch.nn.functional as F
from g_mlp_pytorch import gMLP

class ResMultiConv(nn.Module):
    '''
    基本的残差多卷积模块，2层3x3卷积加残差
    '''
    def __init__(self, channels):
        super(ResMultiConv, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


class GLAM(nn.Module):
    def __init__(self, in_channels, seq_len, feature_dim):
        super(GLAM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(16)

        self.conv4 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(8)

        self.pool = nn.AdaptiveAvgPool2d((25, 96))  # 可调整目标尺寸
        self.project = nn.Linear(96 * 8, 768)

        self.gmlp = gMLP(
            dim=768,
            depth=1,  # 视你训练模型使用的结构而定
            seq_len=25,
            act=nn.Tanh()
        )

    def forward(self, x):
        """
        x: shape [B, Levels=12, Seq, Dim]，形如 [64, 12, 200, 768]
        """
        B, L, S, D = x.shape

        # 合并 batch 和 level 维度，处理成 CNN 适合的形式
        x = x.view(B, L, S, D)
        x = x.permute(0, 1, 3, 2)  # [B, L, D, S]

        x = self.relu(self.bn1(self.conv1(x)))   # [B, 64, D, S]
        x = self.relu(self.bn2(self.conv2(x)))   # [B, 32, D, S]
        x = self.relu(self.bn3(self.conv3(x)))   # [B, 16, D, S]
        x = self.relu(self.bn4(self.conv4(x)))   # [B, 8, D, S]

        x = self.pool(x)  # [B, 8, 25, 96]
        x = x.permute(0, 2, 1, 3).contiguous()  # [B, 25, 8, 96]
        x = x.view(x.size(0), x.size(1), -1)    # [B, 25, 8*96] → [B, 25, 768]
        x = self.project(x)                     # [B, 25, 128]

        # # 动态初始化 gMLP
        # if self.gmlp is None:
        #     seq_len = x.size(1)
        #     dim = x.size(2)
        #     self.gmlp = gMLP(
        #         dim=dim,
        #         depth=1,
        #         seq_len=seq_len,
        #         act=nn.Tanh()
        #     ).to(x.device)

        x = self.gmlp(x)  # [B, seq_len, dim]
        x = x.mean(dim=1)  # Pooling → [B, dim]

        return x
