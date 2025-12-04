import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import UNet
from dataset import DiscreteMNIST

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'


# Forward process: convert image tokens to masked tokens
def forward_process(tokens_0, mask):
    """
    Masked diffusion forward process:
    - tokens_0: original tokens [B, H, W] with values {0, 1}
    - mask: binary mask [B, H, W], 1=preserve pixel, 0=mask pixel
    Returns:
        tokens_t: masked tokens [B, H, W] with values {0, 1, 2}
                  where 2 = MSK (masked pixel)
    """
    tokens_t = tokens_0.clone()
    tokens_t[mask == 0] = 2  # Set masked pixels to MSK token (2)
    return tokens_t

# Training step
def train_step(model, tokens_0):
    """
    Training step for masked diffusion with cross-entropy loss.
    
    Args:
        model: UNet model that takes tokens [B, H, W] and timestep [B], outputs logits [B, 2, H, W]
        tokens_0: original tokens [B, H, W] with values {0, 1}
    """
    batch_size = tokens_0.shape[0]
    
    # Sample random timestep (0 to 999)
    t = torch.randint(0, 1000, (batch_size,), device=device)

    # Weight for timestep (optional, can remove if not needed)
    w = (1000 - t) / (t + 1)

    # Generate a mask where each entry has probability (t/1000) of being 0 (masked)
    prob_zero = t.float().view(-1, 1, 1) / 1000.0
    random_tensor = torch.rand(batch_size, tokens_0.shape[1], tokens_0.shape[2], device=device)
    mask = (random_tensor > prob_zero).float()  # [B, H, W], 1=preserve, 0=mask
    
    # Forward process: create masked tokens
    tokens_t = forward_process(tokens_0, mask)  # [B, H, W] with values {0, 1, 2}
    
    # Model predicts logits for {0, 1} at each pixel
    logits = model(tokens_t, t)  # [B, 2, H, W]
    
    # Cross-entropy loss only on masked tokens
    # Reshape for cross-entropy: [B, 2, H, W] -> [B*H*W, 2] and [B, H, W] -> [B*H*W]
    masked_positions = (mask == 0)  # [B, H, W], True where masked
    if masked_positions.sum() == 0:
        # No masked tokens, return zero loss
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, 2)  # [B*H*W, 2]
    targets_flat = tokens_0.reshape(-1).long()  # [B*H*W]
    mask_flat = masked_positions.reshape(-1)  # [B*H*W]
    
    # Compute loss only on masked positions
    loss = F.cross_entropy(logits_flat[mask_flat], targets_flat[mask_flat], reduction='mean')
    
    loss = loss * w

    return loss

# Main training
def train():
    
    # Model - token-based UNet
    model = UNet(image_size=28, base_channels=32, token_embed_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # DiscreteMNIST dataset (returns [1, 28, 28] with values {0, 1})
    dataset = DiscreteMNIST(root='./data', train=True, download=True, normalize_to_minus1_1=False)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    # Training loop
    num_epochs = 1
    model.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        running_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, images in enumerate(pbar):
            # DiscreteMNIST returns [B, 1, 28, 28] with values {0, 1}
            # Convert to [B, 28, 28] tokens by squeezing channel dimension
            tokens = images.squeeze(1).to(device)  # [B, 28, 28] with values {0, 1}
            
            # Training step
            optimizer.zero_grad()
            loss = train_step(model, tokens)
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

