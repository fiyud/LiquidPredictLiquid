from typing import Optional, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicSnake(nn.Module):
    def __init__(self, channels: int, init_alpha: float = 0.5, clamp_min: float = 0.1, clamp_max: float = 10.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((1, channels, 1, 1), float(init_alpha)))
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.alpha.clamp(self.clamp_min, self.clamp_max)
        return x + torch.sin(a * x) ** 2 / (a + 1e-6)


def _make_act(channels: int, act_type: Literal['relu', 'silu', 'snake'] = 'relu') -> nn.Module:
    if act_type == 'relu':
        return nn.ReLU(inplace=True)
    if act_type == 'silu':
        return nn.SiLU(inplace=True)
    if act_type == 'snake':
        return DynamicSnake(channels)
    raise ValueError(f'Unsupported act_type: {act_type}')

class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 mid_channels: Optional[int] = None,
                 act_type: Literal['relu', 'silu', 'snake'] = 'relu'):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            _make_act(mid_channels, act_type),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            _make_act(out_channels, act_type),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 act_type: Literal['relu', 'silu', 'snake'] = 'relu'):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, act_type=act_type)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling rồi DoubleConv; hỗ trợ bilinear hoặc transposed conv (tùy chọn activation)"""
    def __init__(self, in_channels: int, out_channels: int,
                 bilinear: bool = True,
                 act_type: Literal['relu', 'silu', 'snake'] = 'relu'):
        super().__init__()

        self.bilinear = bilinear
        self.act_type = act_type

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, act_type=act_type)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, act_type=act_type)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # x1: feature từ tầng dưới; x2: skip-connection
        x1 = self.up(x1)

        # Căn lề cho khớp kích thước
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# =========================
# Positional Encoding 2D cho Transformer
# =========================
class PositionalEncoding2D(nn.Module):
    """
    Sinusoidal 2D PE, ánh xạ vào cùng số kênh với tensor đầu vào.
    """
    def __init__(self, channels: int):
        super().__init__()
        assert channels % 4 == 0, "PositionalEncoding2D: channels phải chia hết cho 4"
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        b, c, h, w = x.shape
        device = x.device
        c_quarter = c // 4

        y_pos = torch.arange(h, device=device).float().unsqueeze(1).repeat(1, w)  # (H,W)
        x_pos = torch.arange(w, device=device).float().unsqueeze(0).repeat(h, 1)  # (H,W)

        div_term = torch.exp(
            torch.arange(0, c_quarter, device=device).float() *
            (-torch.log(torch.tensor(10000.0, device=device)) / c_quarter)
        )

        pe_y = torch.zeros(h, w, 2 * c_quarter, device=device)
        pe_x = torch.zeros(h, w, 2 * c_quarter, device=device)

        pe_y[..., 0:c_quarter] = torch.sin(y_pos.unsqueeze(-1) * div_term)
        pe_y[..., c_quarter:2*c_quarter] = torch.cos(y_pos.unsqueeze(-1) * div_term)
        pe_x[..., 0:c_quarter] = torch.sin(x_pos.unsqueeze(-1) * div_term)
        pe_x[..., c_quarter:2*c_quarter] = torch.cos(x_pos.unsqueeze(-1) * div_term)

        pe = torch.cat([pe_y, pe_x], dim=-1).permute(2, 0, 1).unsqueeze(0)  # (1,C,H,W)
        return x + pe


# =========================
# C3STR (giữ nguyên cấu trúc, thêm PE + dropout)
# =========================
class C3STR(nn.Module):
    """
    1x1 conv -> BN/ReLU -> +PE -> TransformerEncoder (chuỗi HW) -> 1x1 conv + residual.
    Giữ H,W; đổi kênh in_channels -> out_channels.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 n_layers: int = 2, n_heads: int = 8, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        embed_dim = out_channels

        self.proj_in = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

        self.posenc = PositionalEncoding2D(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * mlp_ratio,
            batch_first=True,
            activation="gelu",
            norm_first=True,
            dropout=dropout,          # thêm dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.proj_out = nn.Sequential(
            nn.Conv2d(embed_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_in = x

        x = self.proj_in(x)                      # B, C, H, W  (C=out_channels)
        x = self.posenc(x)                       # + positional encoding 2D
        x_seq = x.flatten(2).transpose(1, 2)     # (B, HW, C)
        x_seq = self.transformer(x_seq)
        x = x_seq.transpose(1, 2).reshape(b, x.shape[1], h, w)
        x = self.proj_out(x)

        x = x + self.shortcut(x_in)
        return self.act(x)


class DownC3STR(nn.Module):
    """Downscaling với MaxPool rồi C3STR."""
    def __init__(self, in_channels: int, out_channels: int,
                 n_layers: int = 2, n_heads: int = 8, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.c3str = C3STR(in_channels, out_channels,
                           n_layers=n_layers, n_heads=n_heads, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        return self.c3str(x)
