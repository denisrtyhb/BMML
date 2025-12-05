import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
from model import UNet  # Assuming you have your UNet in model.py

# --- 1. The Masking Logic ---
def get_mask(x_0, t, device):
    """
    Generates a binary mask based on timestep t.
    t=0    -> mask is all 1s (keep everything)
    t=1000 -> mask is all 0s (hide everything)
    """
    batch_size, c, h, w = x_0.shape
    
    # Probability of a pixel being MASKED (0)
    # Linear schedule: at t=0, prob=0. At t=1000, prob=1.
    mask_prob = t.float() / 1000.0
    mask_prob = mask_prob.view(-1, 1, 1, 1) # [B, 1, 1, 1]
    
    # Generate random noise for decision
    rand_tensor = torch.rand(batch_size, 1, h, w, device=device)
    
    # If rand > prob, we keep the pixel (1). Otherwise mask it (0).
    mask = (rand_tensor > mask_prob).float()
    return mask

def forward_process(x_0, mask):
    """
    Apply the mask.
    Masked pixels become 0 (gray in 0-1 range, or -1 in -1to1 range).
    Let's assume input is [-1, 1]. We map masked regions to -1 (black/background).
    """
    # x_0 is [-1, 1]. 
    # masked regions (0 in mask) should become -1 or 0.
    # Let's just multiply. 0 * pixel = 0.
    return x_0 * mask

# --- 2. The Training Step ---
def train_step(model, x_0, device):
    batch_size = x_0.shape[0]
    
    # 1. Sample random timestep t ~ U(0, 1000)
    t = torch.randint(0, 1000, (batch_size,), device=device)
    
    # 2. Create Mask
    mask = get_mask(x_0, t, device)
    
    # 3. Corrupt Image (Forward Process)
    x_t = forward_process(x_0, mask)
    
    # 4. Model Prediction
    # Model sees masked image and knows t. It tries to guess x_0.
    pred_x_0 = model(x_t, t, mask)
    
    # 5. Loss Calculation (The Fix)
    # We compare prediction to original.
    # CRITICAL: We DO NOT weight by (1000-t). We treat all steps equally.
    
    # Option A: Loss on everything (Simple)
    # loss = ((pred_x_0 - x_0) ** 2).mean()
    
    # Option B: Loss only on masked regions (More precise for MDLM)
    # We only care if the model fills in the BLANKS correctly.
    loss_elementwise = (pred_x_0 - x_0) ** 2
    loss_masked = loss_elementwise * (1 - mask) # Only count loss where mask is 0
    
    # Avoid division by zero if mask is all 1s
    num_masked_pixels = (1 - mask).sum()
    if num_masked_pixels > 0:
        loss = loss_masked.sum() / num_masked_pixels
    else:
        loss = loss_elementwise.mean() * 0 # No masked pixels, no loss
        
    return loss

# --- 3. Main Loop ---
def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = UNet(in_channels=1, out_channels=1, base_channels=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # MNIST Loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]), # Map to [-1, 1]
    ])
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    epochs = 5
    
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for x, _ in pbar:
            x = x.to(device)
            optimizer.zero_grad()
            loss = train_step(model, x, device)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())
            
        # Quick visualization at end of epoch
        with torch.no_grad():
            # Create a fully masked dummy input
            dummy_mask = torch.zeros(8, 1, 28, 28).to(device)
            dummy_input = torch.zeros(8, 1, 28, 28).to(device) # All black
            t_input = torch.full((8,), 999, device=device) # t=999
            
            # Prediction
            reconstruction = model(dummy_input, t_input, dummy_mask)
            
            # Save
            grid = make_grid(reconstruction, nrow=4, normalize=True)
            save_image(grid, f"epoch_{epoch}_sample.png")
            
    torch.save(model.state_dict(), "masked_diffusion.pth")
    print("Done!")

if __name__ == "__main__":
    train()