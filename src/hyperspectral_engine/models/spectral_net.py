"""1D卷积 + Self-Attention 多任务联合回归网络

针对GF-5卫星330波段高光谱反射率向量，
精准反演土壤中镉(Cd)、铅(Pb)、砷(As)的绝对浓度 (mg/kg)。

网络设计特点:
    - 轻量化深度可分离1D卷积，提取全波段局部特征
    - 多头自注意力，捕获长程光谱依赖
    - 多任务头共享表征，独立输出Cd/Pb/As
    - 参数总量 < 500K，适合ONNX Runtime低延迟推理
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class DepthwiseSeparableConv1d(nn.Module):
    """深度可分离1D卷积，大幅降低参数量"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(
            in_channels, out_channels, kernel_size=1, bias=False
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class MultiHeadSpectralAttention(nn.Module):
    """光谱维度多头自注意力

    输入形状 [B, C, L] -> 输出 [B, C, L]
    C=通道数(特征维), L=光谱长度(330)
    """

    def __init__(self, channels: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, L)
        q, k, v = qkv.unbind(dim=1)

        q = q.transpose(-1, -2)
        v = v.transpose(-1, -2)

        attn = torch.matmul(q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(-1, -2).contiguous()
        out = out.reshape(B, C, L)
        return self.proj(out)


class SpectralEncoderBlock(nn.Module):
    """光谱编码块: 可分离卷积 + 自注意力 + 残差"""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 5,
        num_heads: int = 4,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = DepthwiseSeparableConv1d(
            channels, channels, kernel_size, padding=padding
        )
        self.conv2 = DepthwiseSeparableConv1d(
            channels, channels, kernel_size, padding=padding
        )
        self.attn = MultiHeadSpectralAttention(channels, num_heads=num_heads)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attn(x)
        return identity + self.residual_scale * x


class MultiTaskRegressionHead(nn.Module):
    """多任务回归头

    共享底层表征，独立预测Cd/Pb/As三种重金属浓度
    """

    def __init__(self, in_features: int, hidden_dim: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        self.cd_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),
        )
        self.pb_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),
        )
        self.as_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        shared = self.shared(x)
        return {
            "Cd": self.cd_head(shared).squeeze(-1),
            "Pb": self.pb_head(shared).squeeze(-1),
            "As": self.as_head(shared).squeeze(-1),
        }


class HyperspectralInversionNet(nn.Module):
    """高光谱污染反演主干网络

    输入: [B, 1, 330] 原始反射率向量
    输出: {"Cd": [B], "Pb": [B], "As": [B]} 浓度 (mg/kg)
    """

    def __init__(
        self,
        in_channels: int = 1,
        spectral_length: int = 330,
        base_channels: int = 32,
        num_blocks: int = 3,
        num_heads: int = 4,
    ):
        super().__init__()
        self.spectral_length = spectral_length

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        current_channels = base_channels
        current_length = spectral_length // 4

        self.encoder_blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.encoder_blocks.append(
                SpectralEncoderBlock(
                    current_channels,
                    kernel_size=5,
                    num_heads=num_heads,
                )
            )
            if i < num_blocks - 1:
                out_ch = current_channels * 2
                self.encoder_blocks.append(
                    DepthwiseSeparableConv1d(
                        current_channels, out_ch, kernel_size=3, stride=2, padding=1
                    )
                )
                current_channels = out_ch
                current_length = (current_length + 1) // 2

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.reg_head = MultiTaskRegressionHead(current_channels, hidden_dim=128)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        assert x.shape[2] == self.spectral_length, (
            f"Expected spectral length {self.spectral_length}, got {x.shape[2]}"
        )

        x = self.stem(x)
        for block in self.encoder_blocks:
            x = block(x)
        pooled = self.global_pool(x).squeeze(-1)
        return self.reg_head(pooled)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(
    spectral_length: int = 330,
    base_channels: int = 32,
    num_blocks: int = 3,
    num_heads: int = 4,
    pretrained_path: str = None,
    device: str = "cpu",
) -> HyperspectralInversionNet:
    """构建模型实例

    Args:
        spectral_length: 光谱通道数 (GF-5=330)
        base_channels: 基础通道数
        num_blocks: 编码块数量
        num_heads: 注意力头数
        pretrained_path: 预训练权重路径
        device: 计算设备

    Returns:
        HyperspectralInversionNet
    """
    model = HyperspectralInversionNet(
        in_channels=1,
        spectral_length=spectral_length,
        base_channels=base_channels,
        num_blocks=num_blocks,
        num_heads=num_heads,
    )
    if pretrained_path and pretrained_path != "":
        state = torch.load(pretrained_path, map_location=device, weights_only=True)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
