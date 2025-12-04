import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from model import UNet
from dataset import DiscreteMNIST
import sys
import numpy as np
from PIL import Image

MODEL_PATH = sys.argv[1]
device = 'cpu'
seed = 42
torch.manual_seed(seed)

def forward_process(tokens_0, mask):
    """
    Convert original tokens to masked tokens.
    tokens_0: [B, H, W] with values {0, 1}
    mask: [B, H, W] with values {0, 1}, 1=preserve, 0=mask
    Returns: tokens_t [B, H, W] with values {0, 1, 2} where 2=MSK
    """
    tokens_t = tokens_0.clone()
    print(tokens_t.shape, mask.shape)
    tokens_t[mask.float() == 1] = 2  # Set masked pixels to MSK token (2)
    return tokens_t

def denoise(model, tokens_t, t):
    """
    Denoise by predicting logits for masked tokens.
    tokens_t: [B, H, W] with values {0, 1, 2}
    t: [B] timestep
    Returns: logits [B, 2, H, W] for classes {0, 1}
    """
    model.eval()
    with torch.no_grad():
        logits = model(tokens_t, t)  # [B, 2, H, W]

    return logits

def logits_to_tokens(logits):
    """Convert logits to predicted tokens using argmax"""
    return logits.argmax(dim=1)  # [B, H, W] with values {0, 1}

def save_as_image(tokens, path):
    """Save tokens as image (0=black, 1=white)"""
    tokens = tokens.cpu().numpy()
    tokens = tokens.squeeze()
    tokens = tokens * 255.0
    tokens = tokens.astype(np.uint8)
    image = Image.fromarray(tokens, mode='L')
    image.save(path)

def denoise_one_step_based_on_mask(model, tokens_before, t, num_to_unmask=None):
    """
    Denoise one step, optionally unmasking some pixels.
    tokens_before: [H, W] with values {0, 1, 2}
    mask: [H, W] with values {0, 1}, 1=preserve, 0=mask
    Returns: tokens_after [H, W], updated_mask [H, W]
    """
    assert tokens_before.shape == (28, 28), f"Before shape: {tokens_before.shape}"
    
    # Get predictions
    logits = denoise(model, tokens_before.unsqueeze(0), t.unsqueeze(0))[0]  # [2, H, W]
    pred_tokens = logits_to_tokens(logits.unsqueeze(0))[0]  # [H, W]
    
    tokens_after = tokens_before.clone()
    
    if num_to_unmask is not None:
        # Unmask some pixels
        masked_candidates = (tokens_before == 2).nonzero(as_tuple=False)  # [N, 2]
        if len(masked_candidates) > 0:
            num_to_unmask = min(num_to_unmask, len(masked_candidates))
            unmask_indices = masked_candidates[torch.randperm(len(masked_candidates))[:num_to_unmask]]
            
            for idx in unmask_indices:
                h, w = idx[0].item(), idx[1].item()
                tokens_after[h, w] = pred_tokens[h, w]
    else:
        # Update all masked positions
        tokens_after[tokens_before == 2] = pred_tokens[tokens_before == 2]
    
    return tokens_after

def denoise_full_trajectory_based_on_mask(model, tokens_start, t, num_steps):
    """
    Denoise over multiple steps, gradually unmasking pixels.
    Returns: trajectory [num_steps, H, W] of tokens at each step
    """
    tokens = tokens_start.clone()
    trajectory = []
    
    for i in range(num_steps-1, -1, -1):
        num_masked = (tokens == 2).sum().item()
        if num_masked > 0:
            num_to_unmask = num_masked // (i + 1)
            tokens = denoise_one_step_based_on_mask(
                model, tokens, t, num_to_unmask=num_to_unmask)
        trajectory.append(tokens.clone())
    
    return torch.stack(trajectory, dim=0)  # [num_steps, H, W]

def sanity_check_denosing(num_samples=16):
    """Test denoising on real images"""
    # Load model
    model = UNet.load_checkpoint(MODEL_PATH).to(device)
    model.eval()
    
    # Load DiscreteMNIST dataset
    dataset = DiscreteMNIST(root='./data', train=False, download=True, normalize_to_minus1_1=False)
    dataloader = DataLoader(dataset, batch_size=num_samples, shuffle=True)
    
    # Get samples
    images = next(iter(dataloader))  # [B, 1, 28, 28] with values {0, 1}
    tokens_0 = images.squeeze(1).to(device)  # [B, 28, 28] with values {0, 1}
    
    # Apply masking (like in training)
    t = torch.randint(1, 1000, (num_samples,), device=device)
    prob_zero = t.float().view(-1, 1, 1) / 1000.0
    random_tensor = torch.rand(num_samples, 28, 28, device=device)
    mask = (random_tensor > prob_zero).float()  # [B, H, W], 1=preserve, 0=mask
    
    print(f"Mask density: {mask.mean(axis=[1,2])}")
    
    # Forward process: create masked tokens
    tokens_t = forward_process(tokens_0, mask)  # [B, H, W] with values {0, 1, 2}
    
    # Denoise
    logits = denoise(model, tokens_t, t)  # [B, 2, H, W]
    tokens_pred = logits_to_tokens(logits)  # [B, H, W] with values {0, 1}
    
    # Prepare for visualization: convert tokens to images [B, 1, H, W]
    images_viz = tokens_0.unsqueeze(1).float()  # [B, 1, H, W]
    masked_viz = tokens_t.unsqueeze(1).float()  # [B, 1, H, W] (2 will show as white)
    masked_viz[masked_viz == 2] = 0.5  # Show MSK as gray
    denoised_viz = tokens_pred.unsqueeze(1).float()  # [B, 1, H, W]
    
    # Create grid: original | masked | denoised
    grid = torch.cat([images_viz, masked_viz, denoised_viz], dim=0)
    grid = make_grid(grid, nrow=num_samples, padding=2)
    
    # Plot and save
    plt.figure(figsize=(15, 5))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy().squeeze(), cmap='gray')
    plt.axis('off')
    plt.title('Original | Masked (gray=MSK) | Denoised')
    plt.tight_layout()
    plt.savefig('sanity_check_denosing.png', dpi=150, bbox_inches='tight')
    print(f"Saved sanity check to sanity_check_denosing.png")

def sample_images(model, num_samples=1, num_steps=10):
    """
    Generate samples from scratch.
    Returns: trajectory [num_steps, num_samples, 1, 28, 28]
    """
    trajectories = []
    for sample in range(num_samples):
        # Start with all tokens masked (all MSK)
        tokens_start = torch.full((28, 28), 2, device=device, dtype=torch.long)  # All MSK
        t = torch.tensor(999, device=device, dtype=torch.long)
        
        trajectory = denoise_full_trajectory_based_on_mask(
            model, tokens_start, t, num_steps=num_steps)
        # trajectory: [num_steps, H, W]
        trajectories.append(trajectory)
    
    # Stack: [num_samples, num_steps, H, W] -> [num_steps, num_samples, 1, H, W]
    trajectories = torch.stack(trajectories, dim=0)  # [num_samples, num_steps, H, W]
    trajectories = trajectories.permute(1, 0, 2, 3)  # [num_steps, num_samples, H, W]
    trajectories = trajectories.unsqueeze(2)  # [num_steps, num_samples, 1, H, W]
    return trajectories

def sanity_check_sampling(num_samples=3, num_steps=10):
    """Test sampling from scratch"""
    model = UNet.load_checkpoint(MODEL_PATH).to(device)
    model.eval()
    
    samples = sample_images(model, num_samples=num_samples, num_steps=num_steps)
    # samples: [num_steps, num_samples, 1, 28, 28] with values {0, 1, 2}
    
    # Convert to float for visualization (MSK=2 -> gray=0.5)
    samples_viz = samples.float()
    samples_viz[samples_viz == 2] = 0.5  # Show MSK as gray
    
    # Rearrange: [num_steps, num_samples, 1, H, W] -> [num_samples, num_steps, 1, H, W]
    samples_viz = samples_viz.permute(1, 0, 2, 3, 4)  # [num_samples, num_steps, 1, H, W]
    # Flatten: [num_samples * num_steps, 1, H, W]
    samples_viz = samples_viz.reshape(num_samples * num_steps, 1, 28, 28)
    
    # Make grid: num_samples rows, num_steps columns
    grid = make_grid(samples_viz, nrow=num_steps, padding=2)
    plt.figure(figsize=(2 * num_steps, 2 * num_samples))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy().squeeze(), cmap='gray')
    plt.axis('off')
    plt.title('Sampling Trajectories (Each Row: Time Steps for One Sample)')
    plt.tight_layout()
    plt.savefig('sanity_check_sampling.png', dpi=150, bbox_inches='tight')
    print(f"Saved sanity check to sanity_check_sampling.png")

if __name__ == '__main__':
    sanity_check_denosing()
    sanity_check_sampling()
