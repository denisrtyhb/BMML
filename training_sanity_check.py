import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from model import UNet
import sys

MODEL_PATH = sys.argv[1]

def forward_process(x_0, t, mask):
    """Apply masking: x_t = mask * x_0"""
    return mask * x_0

def denoise(model, x_t, t, mask, device):
    """Denoise by predicting original image"""
    model.eval()
    with torch.no_grad():
        pred_x_0 = model(x_t, t, mask)
    return pred_x_0

def sanity_check(model_path='checkpoints/simpliest_unet.pth', num_samples=8):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    model = UNet.load_checkpoint(model_path)
    # Load dataset
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=num_samples, shuffle=True)
    
    # Get samples
    images, _ = next(iter(dataloader))
    images = images.to(device)  # [B, 1, 28, 28]
    
    # Apply masking (like in training)
    t = torch.full((num_samples,), 999, device=device)  # High timestep = more masking
    prob_zero = t.float().view(-1, 1, 1, 1) / 1000.0
    random_tensor = torch.rand(num_samples, 1, 28, 28, device=device)
    mask = (random_tensor > prob_zero).float()
    
    # Forward process: create masked images
    masked = forward_process(images, t, mask)
    
    # Denoise
    denoised = denoise(model, masked, t, mask, device)
    
    # Prepare for visualization (denormalize to [0, 1])
    images_viz = (images + 1) / 2.0
    masked_viz = (masked + 1) / 2.0
    denoised_viz = torch.clamp((denoised + 1) / 2.0, 0, 1)
    
    # Create grid: original | masked | denoised
    grid = torch.cat([images_viz, masked_viz, denoised_viz], dim=0)
    grid = make_grid(grid, nrow=num_samples, padding=2)
    
    # Plot and save
    plt.figure(figsize=(15, 5))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap='gray')
    plt.axis('off')
    plt.title('Original | Masked | Denoised')
    plt.tight_layout()
    plt.savefig('sanity_check.png', dpi=150, bbox_inches='tight')
    print(f"Saved sanity check to sanity_check.png")

if __name__ == '__main__':
    sanity_check()
