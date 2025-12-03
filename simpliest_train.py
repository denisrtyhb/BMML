import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# Simple UNet for masked diffusion
from model import UNet

# Forward process for masked diffusion
def forward_process(x_0, t, mask):
    """
    Masked diffusion forward process:
    - t=0: x_t = x_0 (original image)
    - t=1: x_t = mask * x_0 (masked pixels set to 0, preserved pixels keep original)
    
    Args:
        x_0: original image [B, C, H, W]
        t: timestep [B] (0 to 999, integer indices)
        mask: binary mask [B, 1, H, W], 1=preserve pixel, 0=mask pixel (set to 0)
    Returns:
        x_t: masked image [B, C, H, W] where masked pixels are zeros
    """
    return mask * x_0

# Training step
def train_step(model, x_0, device):
    """
    Training step for masked diffusion.
    
    Args:
        model: UNet model that takes (masked_image, timestep, mask) and predicts original image
        x_0: original image [B, C, H, W]
        mask: binary mask [B, 1, H, W], 1=preserve, 0=mask
        device: device to use
    """
    batch_size = x_0.shape[0]
    
    # Sample random timestep (0 to 999)
    t = torch.randint(0, 1000, (batch_size,), device=device)

    # Generate a mask where each entry has probability (t/1000) of being 0, vectorized
    prob_zero = t.float().view(-1, 1, 1, 1) / 1000.0
    random_tensor = torch.rand(batch_size, 1, x_0.shape[2], x_0.shape[3], device=device)
    mask = (random_tensor > prob_zero).float()
    
    x_t = forward_process(x_0, t, mask)  # [B, C, H, W]
    
    pred_x_0 = model(x_t, t, mask)  # [B, C, H, W]
    
    loss = F.mse_loss(pred_x_0, x_0)
    
    return loss

# Main training
def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Model
    model = UNet(in_channels=1, out_channels=1, base_channels=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # MNIST dataset
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),  # Normalize to [-1, 1]
        transforms.v2.GaussianNoise(mean=0.0, std=0.1),
    ])
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    # Training loop
    num_epochs = 10
    model.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        running_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(device)  # [B, 1, 28, 28]
            # Training step
            optimizer.zero_grad()
            loss = train_step(model, images, device)
            loss.backward()
            optimizer.step()
            
            loss_val = loss.item()
            epoch_loss += loss_val
            running_loss += loss_val
            num_batches += 1
            pbar.set_postfix({'loss': f'{loss_val:.4f}'})

            # Print every 100 iterations
            if (batch_idx + 1) % 100 == 0:
                avg_running_loss = running_loss / 100
                print(f"Epoch {epoch+1}, Iteration {batch_idx+1}, Average Loss (last 100): {avg_running_loss:.4f}")
                running_loss = 0.0  # Reset running loss
        
        avg_loss = epoch_loss / num_batches
        print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}')
    
    print("Training complete!")
    model.save_checkpoint('checkpoints/simpliest_unet.pth')

if __name__ == '__main__':
    train()

