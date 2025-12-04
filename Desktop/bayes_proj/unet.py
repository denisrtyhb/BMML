import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, time_emb_dim=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if time_emb_dim:
            self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        else:
            self.time_mlp = None

    def forward(self, x, t=None):
        x = F.relu(self.bn1(self.conv1(x)))
        
        if self.time_mlp is not None and t is not None:
            time_emb = self.time_mlp(t)
            # Broadcast time embedding to spatial dimensions: [B, C] -> [B, C, 1, 1]
            x = x + time_emb[:, :, None, None]
            
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels, time_emb_dim)

    def forward(self, x, t):
        x = self.maxpool(x)
        x = self.conv(x, t)
        return x

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        # Use bilinear upsampling (learnable ConvTranspose can sometimes create checkerboard artifacts)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # We concatenate x1 (upsampled) and x2 (skip connection), so input channels are doubled relative to previous block output
        # But we pass the specific in_channels required
        self.conv = DoubleConv(in_channels, out_channels, time_emb_dim)

    def forward(self, x1, x2, t):
        # x1: Input from previous layer (to be upsampled)
        # x2: Skip connection from Down path
        x1 = self.up(x1)
        
        # --- FIX FOR SIZE MISMATCH ---
        # If input size is odd (e.g. 7x7), maxpool makes it 3x3. 
        # Upsample makes 3x3 back to 6x6. We need padding to match 7x7.
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # -----------------------------

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x, t)
        return x

class UNet(nn.Module):
    def __init__(self, img_channels=1):
        super().__init__()
        time_dim = 128
        
        # Time Embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU()
        )

        # Initial Conv
        # Input channels + 1 for mask channel
        self.inc = DoubleConv(img_channels + 1, 64, time_dim)
        
        # Down Path
        self.down1 = Down(64, 128, time_dim)
        self.down2 = Down(128, 256, time_dim)
        
        # Bottleneck
        self.bot = DoubleConv(256, 512, time_dim)
        
        # Up Path
        # Channels = (skip connection channels) + (upsampled channels)
        self.up1 = Up(512 + 256, 128, time_dim)
        self.up2 = Up(128 + 128, 64, time_dim)
        self.up3 = Up(64 + 64, 64, time_dim)
        
        # Output
        self.outc = nn.Conv2d(64, img_channels, 1)

    def forward(self, x, t, mask_channel): # already masked toekn MUST not change anything
        # x: [B, 1, H, W]
        # mask_channel: [B, 1, H, W]
        x = torch.cat([x, mask_channel], dim=1)
        
        t = self.time_mlp(t)
        
        x1 = self.inc(x, t)        # -> 64 ch
        x2 = self.down1(x1, t)     # -> 128 ch
        x3 = self.down2(x2, t)     # -> 256 ch
        x4 = self.bot(x3, t)       # -> 512 ch
        
        x = self.up1(x4, x3, t)    # 512 + 256 -> 128 ch
        x = self.up2(x, x2, t)     # 128 + 128 -> 64 ch
        x = self.up3(x, x1, t)     # 64 + 64 -> 64 ch
        
        output = self.outc(x)
        return output