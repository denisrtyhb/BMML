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
        self.model = model
        self.image_size = image_size
        self.channels = channels
        self.device = device
        self.num_timesteps = num_timesteps
        
        # Linear noise schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # For sampling
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        
        self.optimizer = None
        
    def q_sample(self, x_start, t, noise=None, mask=None):
        """
        Forward diffusion process with masking.
        Only adds noise to unmasked regions.
        
        Args:
            x_start: Original images [B, C, H, W]
            t: Timesteps [B]
            noise: Optional noise tensor
            mask: Binary mask [B, 1, H, W], 1 = preserve, 0 = corrupt
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        # Add noise to unmasked regions
        if mask is not None:
            # mask: 1 = preserve (keep original), 0 = corrupt (add noise)
            noisy = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
            # Preserve masked regions, corrupt unmasked regions
            x_t = mask * x_start + (1 - mask) * noisy
        else:
            # Standard diffusion (no masking)
            x_t = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        
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
    
    def p_sample(self, x_t, t, mask=None, guidance_scale=1.0):
        """
        Reverse diffusion step (sampling).
        
        Args:
            x_t: Noisy image at timestep t [B, C, H, W]
            t: Timestep [B]
            mask: Binary mask (1 = preserve, 0 = generate) [B, 1, H, W]
        """
        # Predict noise
        predicted_noise = self.model(x_t, t, mask)
        
        # Compute coefficients
        alpha_t = self.alphas[t].view(-1, 1, 1, 1)
        alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        beta_t = self.betas[t].view(-1, 1, 1, 1)
        
        # Predict x_0
        pred_x_start = (x_t - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
        
        # Clip to valid range
        pred_x_start = torch.clamp(pred_x_start, -1, 1)
        
        # Compute mean of posterior
        alpha_cumprod_prev = self.alphas_cumprod_prev[t].view(-1, 1, 1, 1)
        posterior_mean = (
            torch.sqrt(alpha_cumprod_prev) * 
            beta_t / (1.0 - alpha_cumprod_t) * pred_x_start +
            torch.sqrt(alpha_t) * (1.0 - alpha_cumprod_prev) / 
            (1.0 - alpha_cumprod_t) * x_t
        )
        
        # Sample
        posterior_variance_t = self.posterior_variance[t].view(-1, 1, 1, 1)
        noise = torch.randn_like(x_t)
        x_prev = posterior_mean + torch.sqrt(posterior_variance_t) * noise
        
        # Preserve masked regions
        if mask is not None:
            # For masked regions, we want to keep them close to original
            # In practice, we blend the predicted value with the original
            # This is a simplified approach - more sophisticated methods exist
            x_prev = mask * x_t + (1 - mask) * x_prev
        
        return x_prev
    
    def p_sample_loop(self, shape, mask=None, guidance_scale=1.0):
        """
        Full reverse diffusion process.
        
        Args:
            shape: Shape of samples [B, C, H, W]
            mask: Binary mask (1 = preserve, 0 = generate)
        """
        b = shape[0]
        img = torch.randn(shape, device=self.device)
        
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
            mask: Binary mask (1 = preserve, 0 = inpaint) [B, 1, H, W]
            num_steps: Number of sampling steps (default: full process)
            guidance_scale: Guidance scale
        """
        self.model.eval()
        num_steps = num_steps or self.num_timesteps
        
        # Add noise to unmasked regions only
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
        
        Args:
            x_start: Original images [B, C, H, W]
            mask: Binary mask [B, 1, H, W]
        """
        b = x_start.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (b,), device=self.device).long()
        
        # Sample noise
        noise = torch.randn_like(x_start)
        
        # Forward diffusion with masking
        x_t = self.q_sample(x_start, t, noise=noise, mask=mask)
        
        # Predict noise
        predicted_noise = self.model(x_t, t, mask)
        
        # Loss: only on unmasked regions
        if mask is not None:
            loss_mask = (1 - mask).expand_as(noise)
            loss = F.mse_loss(predicted_noise * loss_mask, noise * loss_mask, reduction='sum') / loss_mask.sum()
        else:
            loss = F.mse_loss(predicted_noise, noise)
        
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

