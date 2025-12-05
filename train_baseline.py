import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from unet import UNet
from corrupt_data import CorruptedMNIST
import argparse
import os

parser = argparse.ArgumentParser(description="Train baseline model on CorruptedMNIST")
parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs to train for")
parser.add_argument("--device", type=str, default=None, help="Device to run on ('cuda', 'cpu'); if None, auto-detect")
parser.add_argument("--output_folder", type=str, default="outputs", help="Folder to save outputs/checkpoints")

args = parser.parse_args()

n_epochs = getattr(args, "n_epochs", 10)
output_folder = getattr(args, "output_folder", "outputs")

device = getattr(args, "device", 'cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)


def train():
    
    # DATASET: 10% Corrupted
    OBSERVED_MASK_PCT = 0.1
    dataset = CorruptedMNIST(mask_percentage=OBSERVED_MASK_PCT, train=True)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)
    
    model = UNet().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    for epoch in range(n_epochs):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        
        for x_obs, mask_obs, _ in pbar:
            # x_obs: -1 (masked) or 0/1 (visible)
            x_obs = x_obs.to(device)
            mask_obs = mask_obs.to(device)
            b = x_obs.shape[0]
            
            t_easy = torch.rand(b, device=device) * (1.0 - OBSERVED_MASK_PCT) + OBSERVED_MASK_PCT
            
            # Create extra ma
            mask_probs = t_easy.view(-1, 1, 1, 1).expand(b, 1, 28, 28)
            extra_mask = (torch.rand_like(x_obs) < mask_probs).float()
            
            # Union of Dataset Mask + New Mask
            final_mask = torch.max(mask_obs, extra_mask)
            
            # Create Input (Apply -1 to new holes)
            # We must start from x_obs (which already has -1s)
            # If final_mask is 1, set to -1
            # But x_obs is already -1 where mask_obs is 1.
            # So we just need to blank out the NEWLY masked parts.
            x_input = x_obs.clone()
            # If extra_mask is 1, set to -1
            x_input[extra_mask == 1] = -1.0
            
            # Predict
            pred_logits = model(x_input, t_easy * 1000, final_mask)
            
            # Loss Calculation
            # We only learn on pixels that are:
            # 1. Currently Masked (final_mask == 1)
            # 2. BUT were visible in x_obs (mask_obs == 0)
            learnable_region = (final_mask == 1) & (mask_obs == 0)
            
            # Target is x_obs (0.0 or 1.0)
            loss = F.binary_cross_entropy_with_logits(pred_logits, x_obs, reduction='none')
            loss = (loss * learnable_region).sum() / (learnable_region.sum() + 1e-6)
            
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            
            pbar.set_postfix(loss=loss.item())
        # Save the model every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(output_folder, f"baseline_mnist_epoch{epoch+1}.pth"))
    torch.save(model.state_dict(), os.path.join(output_folder, "baseline_mnist_final.pth"))
    print("Training Done.")

if __name__ == "__main__":
    train()