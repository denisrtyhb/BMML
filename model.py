import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
    UNet architecture for masked diffusion.
    
    Takes masked image x_t, timestep t, and mask as inputs.
    Predicts the original image x_0.
    
    Architecture:
    - Input: masked image [B, C, H, W] + mask [B, 1, H, W] = [B, C+1, H, W]
    - Output: predicted original image [B, C, H, W]
    """
    def __init__(self, in_channels=3, out_channels=3, image_size=32, 
                 base_channels=64, time_embed_dim=128):
        super().__init__()
        self.image_size = image_size
        self.time_embed_dim = time_embed_dim
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.ReLU()
        )
        
        # Initial projection
        self.input_proj = nn.Conv2d(in_channels + 1, base_channels, 3, padding=1)  # +1 for mask
        
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
        
        # Output
        self.output = nn.Conv2d(base_channels, out_channels, 1)
        
    def forward(self, x, timestep, mask=None):
        """
        Forward pass.
        
        Args:
            x: Masked image at timestep t [B, C, H, W]
            timestep: Timestep indices [B] (integer indices 0 to num_timesteps-1)
            mask: Binary mask [B, 1, H, W] (1 = preserve pixel, 0 = mask pixel)
        
        Returns:
            Predicted original image x_0 [B, C, H, W]
        """
        # Time embedding
        t = self.time_mlp(timestep)
        
        # Concatenate mask if provided
        if mask is not None:
            x = torch.cat([x, mask], dim=1)
        else:
            # If no mask provided, create a zero mask (all regions to generate)
            zero_mask = torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3], 
                                   device=x.device, dtype=x.dtype)
            x = torch.cat([x, zero_mask], dim=1)
        
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
        
        # Output
        output = self.output(up3)
        
        return output

class TestUNet(nn.Module):


    def __init__(self):
        super().__init__()
        self.model = UNet(in_channels=3, out_channels=3, base_channels=32)
    def test_unet_forward_no_mask(self):
        batch_size = 1
        h = w = 32
        x = torch.randn(batch_size, 3, h, w)
        t = torch.randint(0, 1000, (batch_size,))
        out = self.model(x, t, None)
        assert out.shape == (batch_size, 3, h, w), "Model output shape mismatch without mask"

    def test_unet_forward_minimal_timestep(self):
        batch_size = 1
        h = w = 32
        x = torch.randn(batch_size, 3, h, w)
        t = torch.zeros((batch_size,), dtype=torch.long)
        mask = torch.ones(batch_size, 1, h, w)
        out = self.model(x, t, mask)
        assert out.shape == (batch_size, 3, h, w)

    def test_unet_input_output_shape(self):
        batch_size = 2
        height = width = 32
        x = torch.randn(batch_size, 3, height, width)
        t = torch.randint(0, 1000, (batch_size,))
        mask = torch.randint(0, 2, (batch_size, 1, height, width)).float()
        out = self.model(x, t, mask)
        assert out.shape == (batch_size, 3, height, width), f"Unexpected output shape {out.shape}"

    def run_all_tests(self):
        self.test_unet_input_output_shape()
        self.test_unet_forward_no_mask()
        self.test_unet_forward_minimal_timestep()
        print("All UNet tests passed.")


if __name__ == "__main__":
    test_unet = TestUNet()
    test_unet.run_all_tests()