import torch
import torch.nn as nn

from .unet_parts import DoubleConv, Down, Up, OutConv, DownC3STR


class UNet(nn.Module):
    def __init__(self,
                 n_channels: int,
                 n_classes: int,
                 bilinear: bool = False,
                 c3str_layers: int = 2,
                 c3str_heads: int = 8,
                 c3str_mlp_ratio: int = 4,
                 c3str_dropout: float = 0.1,
                 act_type_enc: str = 'relu',   # encoder: 'relu' khởi động ổn định
                 act_type_dec: str = 'snake'   # decoder: 'snake' để giàu đặc trưng
                 ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc   = DoubleConv(n_channels, 64,                act_type=act_type_enc)
        self.down1 = Down(64,  128,                            act_type=act_type_enc)
        self.down2 = Down(128, 256,                            act_type=act_type_enc)
        self.down3 = Down(256, 512,                            act_type=act_type_enc)

        factor = 2 if bilinear else 1
        # Bottleneck dùng C3STR (giữ C3STR, chỉ bổ sung P + dropout)
        self.down4 = DownC3STR(
            in_channels=512,
            out_channels=1024 // factor,
            n_layers=c3str_layers,
            n_heads=c3str_heads,
            mlp_ratio=c3str_mlp_ratio,
            dropout=c3str_dropout
        )

        # Decoder (mặc định DynamicSnake để tăng biểu đạt; có thể đổi qua tham số)
        self.up1  = Up(1024, 512 // factor, bilinear, act_type=act_type_dec)
        self.up2  = Up(512,  256 // factor, bilinear, act_type=act_type_dec)
        self.up3  = Up(256,  128 // factor, bilinear, act_type=act_type_dec)
        self.up4  = Up(128,  64,              bilinear, act_type=act_type_dec)
        self.outc = OutConv(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x  = self.up1(x5, x4)
        x  = self.up2(x,  x3)
        x  = self.up3(x,  x2)
        x  = self.up4(x,  x1)
        logits = self.outc(x)
        return logits
