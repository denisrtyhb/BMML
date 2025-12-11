from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as F

class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

class TimeEmbedding(nn.Module):
    def __init__(self, n_channels: int) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        self.act = Swish()
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.view(-1).float()
        half_dim = self.n_channels // 8
        freq_step = math.log(10_000.0) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=t.device) * -freq_step)
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([args.sin(), args.cos()], dim=1)
        emb = self.act(self.lin1(emb))
        return self.lin2(emb)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int, n_groups: int = 32, dropout: float = 0.0) -> None:
        super().__init__()
        if in_channels < n_groups: n_groups = in_channels // 2 if in_channels > 1 else 1
        self.norm1 = nn.GroupNorm(num_groups=n_groups, num_channels=in_channels)
        self.act1 = Swish()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        groups_out = n_groups
        if out_channels < n_groups: groups_out = out_channels // 2 if out_channels > 1 else 1
        self.norm2 = nn.GroupNorm(num_groups=groups_out, num_channels=out_channels)
        self.act2 = Swish()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

        self.time_emb = nn.Linear(time_channels, out_channels)
        self.time_act = Swish()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        time = self.time_emb(self.time_act(t_emb))
        h = h + time[:, :, None, None]
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return h + self.shortcut(x)

class Downsample(nn.Module):
    def __init__(self, n_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, n_channels: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose2d(n_channels, n_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int) -> None:
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_channels)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        return self.res(x, t_emb)

class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int) -> None:
        super().__init__()
        self.res = ResidualBlock(in_channels + out_channels, out_channels, time_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        if x.shape[2:] != skip.shape[2:]:
            diffY = skip.size(2) - x.size(2)
            diffX = skip.size(3) - x.size(3)
            x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x, skip], dim=1)
        return self.res(x, t_emb)

class MiddleBlock(nn.Module):
    def __init__(self, n_channels: int, time_channels: int) -> None:
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, time_channels)
        self.res2 = ResidualBlock(n_channels, n_channels, time_channels)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = self.res1(x, t_emb)
        return self.res2(x, t_emb)

@dataclass
class MDMUNetConfig:
    image_channels: int = 1
    base_channels: int = 64
    channel_multipliers: Tuple[int, ...] = (1, 2, 2, 4)
    num_res_blocks: int = 2
    time_emb_dim_mult: int = 4

class MDMUNet(nn.Module):
    def __init__(self, cfg: MDMUNetConfig = MDMUNetConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        input_channels = cfg.image_channels + 1
        n_channels = cfg.base_channels
        ch_mults = cfg.channel_multipliers
        n_blocks = cfg.num_res_blocks
        time_channels = n_channels * cfg.time_emb_dim_mult

        self.image_proj = nn.Conv2d(input_channels, n_channels, kernel_size=3, padding=1)
        self.time_emb = TimeEmbedding(time_channels)

        down_modules = []
        in_channels = n_channels
        n_resolutions = len(ch_mults)

        for i in range(n_resolutions):
            out_channels = n_channels * ch_mults[i]
            for _ in range(n_blocks):
                down_modules.append(DownBlock(in_channels, out_channels, time_channels))
                in_channels = out_channels
            
            if i < n_resolutions - 1:
                down_modules.append(Downsample(in_channels))

        self.down = nn.ModuleList(down_modules)
        self.middle = MiddleBlock(in_channels, time_channels)

        up_modules = []
        for i in reversed(range(n_resolutions)):
            out_channels = n_channels * ch_mults[i]
            if i < n_resolutions - 1:
                up_modules.append(Upsample(in_channels))
            
            for _ in range(n_blocks):
                up_modules.append(UpBlock(in_channels, out_channels, time_channels))
                in_channels = out_channels

        self.up = nn.ModuleList(up_modules)

        norm_groups = 8 if in_channels % 8 == 0 else (in_channels // 2 if in_channels > 1 else 1)
        self.norm = nn.GroupNorm(num_groups=norm_groups, num_channels=in_channels)
        self.act = Swish()
        self.final = nn.Conv2d(in_channels, cfg.image_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, mask_channel: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x, mask_channel], dim=1)
        t_emb = self.time_emb(t)
        x = self.image_proj(x)
        
        skips = [x]
        for m in self.down:
            x = m(x, t_emb)
            if not isinstance(m, Downsample):
                skips.append(x)

        x = self.middle(x, t_emb)

        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x, t_emb)
            else:
                skip = skips.pop()
                x = m(x, skip, t_emb)

        return self.final(self.act(self.norm(x)))