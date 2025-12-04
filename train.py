import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import os

from unet import UNet
from mdlm_logic import generate_mask, get_loss_weight

# def train():
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     print(f"Device: {device}")
    
#     # 1. Setup Data
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize([0.5], [0.5]) # -1 to 1
#     ])
#     dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
#     loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2)
    
#     # 2. Setup Model
#     model = UNet().to(device)
#     optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
#     epochs = 50
    
#     for epoch in range(epochs):
#         model.train()
#         pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        
#         for x, _ in pbar:
#             x = x.to(device)
#             b = x.shape[0]
            
#             # --- MDLM TRAINING STEP ---
            
#             # 1. Sample continuous time t ~ U(0, 1)
#             # Use a small epsilon to avoid t=0 (no masking)
#             t = torch.rand(b, device=device) * 0.999 + 0.001
            
#             # 2. Create Mask (1=Masked, 0=Visible)
#             # Proportion of 1s is roughly 't'
#             mask = generate_mask(x, t)
            
#             # 3. Apply Mask to Input
#             # If masked, pixel becomes 0 (absorbing state)
#             # Note: since MNIST is -1 to 1, let's use -1 as "black/masked" usually, 
#             # but strictly in MDLM '0' is the [MASK] token. 
#             # Let's just zero it out.
#             x_masked = x * (1 - mask) 
            
#             # 4. Forward Pass
#             # We pass x_masked AND the mask channel so UNet knows what is missing
#             # Input to time embedding is t * 1000 to match old embeddings
#             pred_x0 = model(x_masked, t * 1000, mask)
            
#             # 5. Calculate Loss
#             # We assume the model should predict the original x
#             mse = (pred_x0 - x) ** 2
            
#             # CRITICAL MDLM PART:
#             # We ONLY care about the loss on the MASKED pixels.
#             # (The model just copies the visible ones trivially).
#             loss_masked = mse * mask
            
#             # Weighting from paper (1/t)
#             weights = get_loss_weight(t).view(-1, 1, 1, 1)
#             weighted_loss = loss_masked * weights
            
#             # Average over non-zero elements
#             final_loss = weighted_loss.sum() / (mask.sum() + 1e-6)
            
#             optim.zero_grad()
#             final_loss.backward()
#             optim.step()
            
#             pbar.set_postfix(loss=final_loss.item())
            
#     torch.save(model.state_dict(), "mdlm_mnist_epochs_50.pth")
#     print("Training Done.")

# if __name__ == "__main__":
#     train()

# ... imports ...

# def train():
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     print(f"Device: {device}")
    
#     # 1. Setup Data
#     transform = transforms.Compose([
#         transforms.Resize((28, 28)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5], [0.5]) # -1 to 1
#     ])
#     dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
#     loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2)
    
#     # 2. Setup Model
#     model = UNet().to(device)
#     optim = torch.optim.AdamW(model.parameters(), lr=1e-4) # Keep learning rate
    
#     epochs = 3 # 50 is good
    
#     for epoch in range(epochs):
#         model.train()
#         pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        
#         for x, _ in pbar:
#             x = x.to(device)
#             b = x.shape[0]
            
#             # 1. Sample continuous time t ~ U(0, 1)
#             t = torch.rand(b, device=device)
            
#             # 2. Create Mask (1=Masked, 0=Visible)
#             # We import this from mdlm_logic, or just define it inline
#             mask_probs = t.view(-1, 1, 1, 1).expand(b, 1, 28, 28)
#             mask = (torch.rand_like(x) < mask_probs).float()
            
#             # --- FIX #1: MASK VALUE ---
#             # Previous: x_masked = x * (1 - mask)  -> Sets to 0 (Gray)
#             # New: Set masked pixels to -1 (Black / Background)
#             # If mask is 1, term becomes -1. If mask is 0, term becomes x.
#             x_masked = x * (1 - mask) - 1.0 * mask
            
#             # 3. Forward Pass
#             pred_x0 = model(x_masked, t * 1000, mask)
            
#             # 4. Calculate Loss
#             mse = (pred_x0 - x) ** 2
            
#             # Only loss on masked pixels
#             loss_masked = mse * mask
            
#             # --- FIX #2: UNIFORM WEIGHTING ---
#             # Previous: weighted_loss = loss_masked * (1/t)
#             # New: No weighting (or weight=1). Treat creating structure (t=1)
#             # as just as important as cleaning up (t=0).
            
#             final_loss = loss_masked.sum() / (mask.sum() + 1e-6)
            
#             optim.zero_grad()
#             final_loss.backward()
#             optim.step()
            
#             pbar.set_postfix(loss=final_loss.item())
            
#     torch.save(model.state_dict(), "mdlm_mnist.pth")
#     print("Training Done.")

# if __name__ == "__main__":
#     train()

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from unet import UNet

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # 1. Setup Data - STRICT BINARY (0.0 or 1.0)
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        # Threshold: anything above 0.5 becomes 1.0, else 0.0
        transforms.Lambda(lambda x: (x > 0.5).float()) 
    ])
    
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2)
    
    # 2. Setup Model
    model = UNet().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    epochs = 20 # 20 is enough for binary
    
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        
        for x, _ in pbar:
            x = x.to(device)
            b = x.shape[0]
            
            # 1. Sample continuous time t ~ U(0, 1)
            t = torch.rand(b, device=device)
            
            # 2. Create Mask
            mask_probs = t.view(-1, 1, 1, 1).expand(b, 1, 28, 28)
            mask = (torch.rand_like(x) < mask_probs).float()
            
            # 3. Apply Mask
            # Image is 0 or 1.
            # Masked regions become 0. 
            # (The Mask Channel tells the UNet that these zeros are actually 'holes')
            x_masked = x * (1 - mask)
            
            # 4. Forward Pass
            # Output is "Logits" (Scores), not probabilities yet
            pred_logits = model(x_masked, t * 1000, mask)
            
            # 5. Calculate Loss (Binary Cross Entropy)
            # We compare Logits vs Target (0 or 1)
            # reduction='none' allows us to mask the loss later
            loss = F.binary_cross_entropy_with_logits(pred_logits, x, reduction='none')
            
            # Only learn from masked pixels
            masked_loss = loss * mask
            
            # Average over masked pixels
            final_loss = masked_loss.sum() / (mask.sum() + 1e-6)
            
            optim.zero_grad()
            final_loss.backward()
            optim.step()
            
            pbar.set_postfix(loss=final_loss.item())
            
    torch.save(model.state_dict(), "mdlm_mnist_binary.pth")
    print("Training Done.")

if __name__ == "__main__":
    train()
    