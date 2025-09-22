import torch
import torch.nn as nn

class GeM(nn.Module):
    """Generalized Mean Pooling: mặc định m=3.0; m=1 -> GAP, m->inf -> GMP"""
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p))
        self.eps = eps
    
    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return torch.mean(x, dim=(-2, -1)).pow(1.0 / self.p)  # (B,C)

class VGGMSFeatureExtractor(nn.Module):
    """
    VGG-based Multi-Scale Feature Extractor
    Rút đặc trưng từ các tầng VGG: conv3_x, conv4_x, conv5_x (multi-scale) -> concat -> head -> 1024-d.
    - Đóng băng backbone để ổn định khi chỉ train GRU.
    - Dùng GeM (ổn định hơn GAP) + LayerNorm + MLP nhỏ.
    """
    def __init__(self, input_channels=1, output_features=1024, freeze_backbone=True):
        super().__init__()
        
        # VGG19-like backbone
        # Block 1: 64 channels
        self.block1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 320->160
        )
        
        # Block 2: 128 channels
        self.block2 = nn.Sequeeeeeeeeeeeee
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 160->80
        )
        
        # Block 3: 256 channels (multi-scale feature 1)
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 80->40
        )
        
        # Block 4: 512 channels (multi-scale feature 2)
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 40->20
        )
        
        # Block 5: 512 channels (multi-scale feature 3)
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 20->10
        )
        
        # GeM pooling for each scale
        self.pool = GeM(p=3.0)
        
        # Calculate concatenated dimension: 256 + 512 + 512 = 1280
        concat_dim = 256 + 512 + 512
        
        self.norm = nn.LayerNorm(concat_dim)
        self.head = nn.Sequential(
            nn.Linear(concat_dim, 1024), 
            nn.ReLU(inplace=True),
            nn.Linear(1024, output_features), 
            nn.ReLU(inplace=True),
        )
        
        if freeze_backbone:
            for p in self.parameters():
                if p is not self.norm.weight and p is not self.norm.bias:
                    p.requires_grad = False
            # Chỉ cho phép train head và norm
            for p in self.head.parameters():
                p.requires_grad = True
            for p in self.norm.parameters():
                p.requires_grad = True
        
        self.eval()  # mặc định đặt eval để BN/Dropout cố định khi trích xuất

    @torch.no_grad()
    def forward(self, x):
        # Forward through VGG blocks
        x = self.block1(x)  # 64 channels
        x = self.block2(x)  # 128 channels
        x3 = self.block3(x)  # 256 channels, 40x40
        x4 = self.block4(x3)  # 512 channels, 20x20  
        x5 = self.block5(x4)  # 512 channels, 10x10
        
        # GeM pooling on multi-scale features
        z3 = self.pool(x3)  # (B, 256)
        z4 = self.pool(x4)  # (B, 512)
        z5 = self.pool(x5)  # (B, 512)
        
        # Concatenate multi-scale features
        z = torch.cat([z3, z4, z5], dim=1)  # (B, 1280)
        z = self.norm(z)
        z = self.head(z)  # (B, 1024)
        return z

# khởi tạo VGG extractor (đóng băng backbone, chỉ rút đặc trưng)
# vgg_model = VGGMSFeatureExtractor(input_channels=1, output_features=1024, freeze_backbone=True).to(device)