import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os

class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal positional embeddings for timesteps"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    """Basic convolutional block"""
    def __init__(self, in_channels, out_channels, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        
        if up:
            self.conv1 = nn.Conv2d(2*in_channels, out_channels, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_channels, out_channels, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            self.transform = nn.Conv2d(out_channels, out_channels, 4, 2, 1)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_channels)
        self.bnorm2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x, t, mask=None):
        # Time embedding
        t = self.time_mlp(t)
        t = t.unsqueeze(-1).unsqueeze(-1)
        
        # First conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Add time embedding
        h = h + t
        # Second conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)

class UNet(nn.Module):
    """
    UNet architecture for masked diffusion with token-based input.
    
    Takes tokens [B, H, W] where each token is from {0, 1, MSK}:
    - 0 = pixel value 0
    - 1 = pixel value 1
    - 2 = MSK (masked pixel)
    
    Predicts logits for {0, 1} at each pixel.
    
    Architecture:
    - Input: tokens [B, H, W] with values in {0, 1, 2}
    - Output: logits [B, 2, H, W] for classes {0, 1}
    """
    def __init__(self, image_size=32, base_channels=64, time_embed_dim=128, 
                 token_embed_dim=64):
        super().__init__()
        self.image_size = image_size
        self.time_embed_dim = time_embed_dim
        self.base_channels = base_channels
        self.token_embed_dim = token_embed_dim
        
        # Token embedding: 3 tokens {0, 1, MSK} -> embed_dim
        self.token_embed = nn.Embedding(3, token_embed_dim)  # 0, 1, MSK
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.ReLU()
        )
        
        # Initial projection: token embeddings -> base_channels
        self.input_proj = nn.Conv2d(token_embed_dim, base_channels, 3, padding=1)
        
        # Downsampling
        self.down1 = Block(base_channels, base_channels, time_embed_dim)
        self.down2 = Block(base_channels, base_channels * 2, time_embed_dim)
        self.down3 = Block(base_channels * 2, base_channels * 4, time_embed_dim)
        
        # Bottleneck
        self.bot1 = nn.Conv2d(base_channels * 4, base_channels * 8, 3, padding=1)
        self.bot2 = nn.Conv2d(base_channels * 8, base_channels * 8, 3, padding=1)
        self.bot3 = nn.Conv2d(base_channels * 8, base_channels * 4, 3, padding=1)
        
        # Upsampling
        self.up1 = Block(base_channels * 4, base_channels * 2, time_embed_dim, up=True)
        self.up2 = Block(base_channels * 2, base_channels, time_embed_dim, up=True)
        self.up3 = Block(base_channels, base_channels, time_embed_dim, up=True)
        
        # Output: logits for {0, 1}
        self.output = nn.Conv2d(base_channels, 2, 1)
        
    def forward(self, tokens, timestep):
        """
        Forward pass.
        
        Args:
            tokens: Token tensor [B, H, W] with values in {0, 1, 2}
                   0 = pixel value 0
                   1 = pixel value 1
                   2 = MSK (masked pixel)
            timestep: Timestep indices [B] (integer indices 0 to num_timesteps-1)
        
        Returns:
            Logits for {0, 1} at each pixel [B, 2, H, W]
        """
        # Time embedding
        t = self.time_mlp(timestep)
        
        # Embed tokens: [B, H, W] -> [B, token_embed_dim, H, W]
        x = self.token_embed(tokens.long())  # [B, H, W, token_embed_dim]
        x = x.permute(0, 3, 1, 2)  # [B, token_embed_dim, H, W]
        
        # Initial projection
        x = self.input_proj(x)
        
        # Downsampling
        down1 = self.down1(x, t)
        down2 = self.down2(down1, t)
        down3 = self.down3(down2, t)
        
        # Bottleneck
        bot = F.relu(self.bot1(down3))
        bot = F.relu(self.bot2(bot))
        bot = F.relu(self.bot3(bot))
        
        # Upsampling with skip connections
        # Need to ensure spatial dimensions match for concatenation
        up1 = self.up1(torch.cat((bot, down3), dim=1), t)
        
        # Resize up1 to match down2's spatial dimensions if needed
        if up1.shape[2:] != down2.shape[2:]:
            up1 = F.interpolate(up1, size=down2.shape[2:], mode='bilinear', align_corners=False)
        up2 = self.up2(torch.cat((up1, down2), dim=1), t)
        
        # Resize up2 to match down1's spatial dimensions if needed
        if up2.shape[2:] != down1.shape[2:]:
            up2 = F.interpolate(up2, size=down1.shape[2:], mode='bilinear', align_corners=False)
        up3 = self.up3(torch.cat((up2, down1), dim=1), t)
        
        # Output: logits for {0, 1}
        logits = self.output(up3)  # [B, 2, H, W]
        
        return logits
    
    def save_checkpoint(self, path):
        # Create the directory for the checkpoint if it doesn't exist
        dirpath = os.path.dirname(path)
        if dirpath and not os.path.exists(dirpath):
            os.makedirs(dirpath)
        state_dict = self.state_dict()
        init_args = {
            'image_size': self.image_size,
            'base_channels': self.base_channels,
            'time_embed_dim': self.time_embed_dim,
            'token_embed_dim': self.token_embed_dim
        }
        torch.save({
            'state_dict': state_dict,
            'init_args': init_args
        }, path)
    
    @staticmethod
    def load_checkpoint(path):
        checkpoint = torch.load(path, map_location='cpu')
        model = UNet(**checkpoint['init_args'])
        model.load_state_dict(checkpoint['state_dict'])
        return model

class TestUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = UNet(image_size=32, base_channels=32)
    
    def test_unet_forward(self):
        batch_size = 1
        h = w = 32
        tokens = torch.randint(0, 3, (batch_size, h, w))  # {0, 1, 2}
        t = torch.randint(0, 1000, (batch_size,))
        out = self.model(tokens, t)
        assert out.shape == (batch_size, 2, h, w), f"Expected [B, 2, H, W], got {out.shape}"

    def test_unet_forward_minimal_timestep(self):
        batch_size = 1
        h = w = 32
        tokens = torch.randint(0, 3, (batch_size, h, w))
        t = torch.zeros((batch_size,), dtype=torch.long)
        out = self.model(tokens, t)
        assert out.shape == (batch_size, 2, h, w)

    def test_unet_input_output_shape(self):
        batch_size = 2
        height = width = 32
        tokens = torch.randint(0, 3, (batch_size, height, width))
        t = torch.randint(0, 1000, (batch_size,))
        out = self.model(tokens, t)
        assert out.shape == (batch_size, 2, height, width), f"Unexpected output shape {out.shape}"

    def run_all_tests(self):
        self.test_unet_input_output_shape()
        self.test_unet_forward()
        self.test_unet_forward_minimal_timestep()
        print("All UNet tests passed.")


if __name__ == "__main__":
    test_unet = TestUNet()
    test_unet.run_all_tests()