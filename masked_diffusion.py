import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

class MaskedDiffusionTrainer:
    def __init__(self, model, image_size=32, channels=3, device='cuda', 
                 num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        """
        Masked Diffusion Trainer.
        
        For masked diffusion:
        - t=0: Original image (x_0)
        - t=1: Fully masked image (masked pixels = 0, unmasked pixels = original)
        - Forward process: x_t = (1 - t) * x_0 + t * (mask * x_0)
          where mask: 1 = preserve pixel, 0 = mask pixel
        """
        self.model = model
        self.image_size = image_size
        self.channels = channels
        self.device = device
        self.num_timesteps = num_timesteps
        
        # Time schedule: t goes from 0 to 1
        # We'll use discrete timesteps 0 to num_timesteps-1, normalized to [0, 1]
        self.timesteps = torch.linspace(0, 1, num_timesteps).to(device)
        
        self.optimizer = None
        
    def q_sample(self, x_start, t, mask):
        """
        Forward diffusion process with masking.
        
        At timestep t:
        - t=0: x_t = x_start (original image)
        - t=1: x_t = mask * x_start (fully masked: masked pixels = 0, preserved pixels = original)
        
        Formula: x_t = (1 - t) * x_start + t * (mask * x_start)
                 = ((1 - t) + t * mask) * x_start
        
        Args:
            x_start: Original images [B, C, H, W]
            t: Timestep indices [B] (integer indices 0 to num_timesteps-1)
            mask: Binary mask [B, 1, H, W], 1 = preserve pixel, 0 = mask pixel
        """
        # Convert integer timestep indices to continuous values in [0, 1]
        t_continuous = t.float() / (self.num_timesteps - 1)  # [B]
        t_continuous = t_continuous.view(-1, 1, 1, 1)  # [B, 1, 1, 1]
        
        # Forward process: x_t = ((1 - t) + t * mask) * x_start
        # At t=0: x_t = x_start (full image)
        # At t=1: x_t = mask * x_start (masked image)
        x_t = ((1 - t_continuous) + t_continuous * mask) * x_start
        
        return x_t
    
    def generate_mask(self, batch_size, mask_ratio=0.5, mask_type='random'):
        """
        Generate masks for training/inference.
        
        Args:
            batch_size: Batch size
            mask_ratio: Ratio of pixels to mask (preserve)
            mask_type: 'random', 'center', 'block'
        """
        mask = torch.zeros(batch_size, 1, self.image_size, self.image_size, device=self.device)
        
        if mask_type == 'random':
            # Random pixel masking
            num_pixels = int(self.image_size * self.image_size * mask_ratio)
            for i in range(batch_size):
                flat_mask = torch.zeros(self.image_size * self.image_size, device=self.device)
                indices = torch.randperm(self.image_size * self.image_size, device=self.device)[:num_pixels]
                flat_mask[indices] = 1.0
                mask[i] = flat_mask.view(1, self.image_size, self.image_size)
        
        elif mask_type == 'center':
            # Center block masking
            center_size = int(self.image_size * np.sqrt(mask_ratio))
            start = (self.image_size - center_size) // 2
            end = start + center_size
            mask[:, :, start:end, start:end] = 1.0
        
        elif mask_type == 'block':
            # Random block masking
            block_size = int(self.image_size * np.sqrt(mask_ratio))
            for i in range(batch_size):
                if block_size < self.image_size:
                    x = torch.randint(0, self.image_size - block_size + 1, (1,), device=self.device).item()
                    y = torch.randint(0, self.image_size - block_size + 1, (1,), device=self.device).item()
                    mask[i, :, x:x+block_size, y:y+block_size] = 1.0
                else:
                    mask[i] = 1.0
        
        return mask
    
    def p_sample(self, x_t, t, mask, guidance_scale=1.0):
        """
        Reverse diffusion step (sampling).
        
        The model predicts x_0 (original image) from x_t (masked image at timestep t).
        We then compute x_{t-1} using the forward process formula.
        
        Args:
            x_t: Masked image at timestep t [B, C, H, W]
            t: Timestep indices [B] (integer indices)
            mask: Binary mask [B, 1, H, W], 1 = preserve pixel, 0 = mask pixel
        """
        # Predict original image x_0
        pred_x_start = self.model(x_t, t, mask)
        
        # Clip to valid range
        pred_x_start = torch.clamp(pred_x_start, -1, 1)
        
        # Compute x_{t-1} using forward process formula
        # If t > 0, we go back one step
        # x_{t-1} = ((1 - t_prev) + t_prev * mask) * pred_x_start
        
        # Get previous timestep
        t_prev = torch.clamp(t - 1, min=0).float() / (self.num_timesteps - 1)  # [B]
        t_prev = t_prev.view(-1, 1, 1, 1)  # [B, 1, 1, 1]
        
        # Apply forward process formula to get x_{t-1}
        x_prev = ((1 - t_prev) + t_prev * mask) * pred_x_start
        
        # For preserved pixels (mask=1), we can optionally keep them from x_t
        # This ensures consistency during reverse process
        # x_prev = mask * x_t + (1 - mask) * x_prev
        
        return x_prev
    
    def p_sample_loop(self, shape, mask, guidance_scale=1.0):
        """
        Full reverse diffusion process.
        
        Start from fully masked image (t=num_timesteps-1) and denoise to original (t=0).
        
        At t=1 (fully masked): x_1 = mask * x_0
        - Preserved pixels (mask=1): keep original value (but we don't know x_0 yet)
        - Masked pixels (mask=0): are 0
        
        For generation:
        - If mask has preserved pixels, we need to know their values (inpainting case)
        - If mask is all zeros, we generate from scratch (all pixels start at 0)
        
        Args:
            shape: Shape of samples [B, C, H, W]
            mask: Binary mask [B, 1, H, W], 1 = preserve pixel, 0 = mask pixel
        """
        b = shape[0]
        
        # Start from fully masked image at t=1
        # x_1 = mask * x_0, so for masked pixels (mask=0), x_1 = 0
        # For preserved pixels (mask=1), x_1 = x_0, but we don't know x_0 yet
        # So we start with zeros everywhere, and the model will predict x_0
        img = torch.zeros(shape, device=self.device)
        
        # If we have preserved pixels (for inpainting), we should initialize them
        # But for generation from scratch, mask should be all zeros
        # For now, we'll let the model handle it
        
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling'):
            t = torch.full((b,), i, device=self.device, dtype=torch.long)
            img = self.p_sample(img, t, mask=mask, guidance_scale=guidance_scale)
        
        return img
    
    def sample(self, num_samples=16, mask_ratio=0.5, mask_type='random', guidance_scale=1.0):
        """
        Generate samples.
        
        Args:
            num_samples: Number of samples to generate
            mask_ratio: Ratio of pixels to preserve
            mask_type: Type of mask
            guidance_scale: Guidance scale (currently not used, for future CFG)
        """
        self.model.eval()
        with torch.no_grad():
            shape = (num_samples, self.channels, self.image_size, self.image_size)
            mask = self.generate_mask(num_samples, mask_ratio, mask_type)
            samples = self.p_sample_loop(shape, mask=mask, guidance_scale=guidance_scale)
        return samples
    
    def inpaint(self, x_start, mask, num_steps=None, guidance_scale=1.0):
        """
        Inpainting: generate content for masked regions.
        
        Args:
            x_start: Original image [B, C, H, W]
            mask: Binary mask [B, 1, H, W], 1 = preserve pixel, 0 = inpaint pixel
            num_steps: Number of sampling steps (default: full process)
            guidance_scale: Guidance scale
        """
        self.model.eval()
        num_steps = num_steps or self.num_timesteps
        
        # Start from fully masked version
        t = torch.full((x_start.shape[0],), num_steps - 1, device=self.device, dtype=torch.long)
        x_t = self.q_sample(x_start, t, mask=mask)
        
        # Reverse process
        with torch.no_grad():
            for i in tqdm(reversed(range(0, num_steps)), desc='Inpainting'):
                t = torch.full((x_start.shape[0],), i, device=self.device, dtype=torch.long)
                x_t = self.p_sample(x_t, t, mask=mask, guidance_scale=guidance_scale)
        
        return x_t
    
    def train_step(self, x_start, mask):
        """
        Single training step.
        
        The model learns to predict the original image x_0 from the masked image x_t.
        
        Args:
            x_start: Original images [B, C, H, W]
            mask: Binary mask [B, 1, H, W], 1 = preserve pixel, 0 = mask pixel
        """
        b = x_start.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (b,), device=self.device).long()
        
        # Forward diffusion with masking: get x_t
        x_t = self.q_sample(x_start, t, mask=mask)
        
        # Model predicts original image x_0
        pred_x_start = self.model(x_t, t, mask)
        
        # Loss: MSE between predicted and actual original image
        # We can compute loss only on masked regions (where we need to predict)
        # or on all regions. Let's use all regions for simplicity.
        loss = F.mse_loss(pred_x_start, x_start)
        
        # Alternatively, focus loss on masked regions only:
        # loss_mask = (1 - mask).expand_as(x_start)  # 1 for masked pixels, 0 for preserved
        # loss = F.mse_loss(pred_x_start * loss_mask, x_start * loss_mask, reduction='sum') / (loss_mask.sum() + 1e-8)
        
        return loss
    
    def train(self, dataloader, epochs, lr, save_dir, sample_dir, 
              save_every=10, sample_every=5, start_epoch=0):
        """
        Training loop.
        """
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        self.model.train()
        
        for epoch in range(start_epoch, epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
            for batch_idx, batch in enumerate(pbar):
                if isinstance(batch, (list, tuple)):
                    x_start = batch[0]
                else:
                    x_start = batch
                
                x_start = x_start.to(self.device)
                
                # Normalize to [-1, 1] if needed
                if x_start.max() > 1.0:
                    x_start = x_start / 255.0 * 2.0 - 1.0
                
                # Generate random mask for each batch
                mask = self.generate_mask(x_start.shape[0], mask_ratio=0.5, mask_type='random')
                
                # Training step
                self.optimizer.zero_grad()
                loss = self.train_step(x_start, mask)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_loss = epoch_loss / num_batches
            print(f'Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}')
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': avg_loss,
                }, checkpoint_path)
                print(f'Saved checkpoint to {checkpoint_path}')
            
            # Generate samples
            if (epoch + 1) % sample_every == 0:
                self.model.eval()
                with torch.no_grad():
                    samples = self.sample(num_samples=16, mask_ratio=0.5, mask_type='random')
                    samples_denorm = (samples + 1) / 2.0  # Denormalize to [0, 1]
                    samples_denorm = torch.clamp(samples_denorm, 0, 1)
                    
                    grid = make_grid(samples_denorm.cpu(), nrow=4, padding=2)
                    sample_path = os.path.join(sample_dir, f'samples_epoch_{epoch+1}.png')
                    plt.imsave(sample_path, grid.permute(1, 2, 0).cpu().numpy())
                    print(f'Saved samples to {sample_path}')
                self.model.train()

