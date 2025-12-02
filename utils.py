import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import os

def save_images(images, path, nrow=4):
    """
    Save a grid of images.
    
    Args:
        images: Tensor of images [B, C, H, W] in range [-1, 1] or [0, 1]
        path: Path to save image
        nrow: Number of images per row
    """
    # Ensure images are in [0, 1] range
    if images.min() < 0:
        images = (images + 1) / 2.0
    images = torch.clamp(images, 0, 1)
    
    # Create grid
    grid = make_grid(images.cpu(), nrow=nrow, padding=2, pad_value=1.0)
    
    # Convert to numpy and save
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    plt.imsave(path, grid_np)
    print(f"Saved images to {path}")

def load_checkpoint(checkpoint_path, model, optimizer=None):
    """
    Load checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint
        model: Model to load weights into
        optimizer: Optional optimizer to load state
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint

def save_checkpoint(model, optimizer, epoch, loss, save_path):
    """
    Save checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer to save
        epoch: Current epoch
        loss: Current loss
        save_path: Path to save checkpoint
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, save_path)
    print(f"Saved checkpoint to {save_path}")

def visualize_mask(image, mask, save_path=None):
    """
    Visualize image with mask overlay.
    
    Args:
        image: Image tensor [C, H, W] or [B, C, H, W]
        mask: Mask tensor [1, H, W] or [B, 1, H, W]
        save_path: Optional path to save visualization
    """
    if image.dim() == 4:
        image = image[0]
    if mask.dim() == 4:
        mask = mask[0]
    
    # Denormalize image
    if image.min() < 0:
        image = (image + 1) / 2.0
    image = torch.clamp(image, 0, 1)
    
    # Create overlay
    overlay = image.clone()
    # Red tint for masked (preserved) regions
    overlay[0] = torch.clamp(overlay[0] + mask[0] * 0.3, 0, 1)
    
    # Concatenate original, mask, and overlay
    vis = torch.cat([image, mask.expand_as(image), overlay], dim=2)
    
    if save_path:
        plt.imsave(save_path, vis.permute(1, 2, 0).cpu().numpy())
    else:
        plt.figure(figsize=(12, 4))
        plt.imshow(vis.permute(1, 2, 0).cpu().numpy())
        plt.axis('off')
        plt.tight_layout()
        plt.show()

