import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from unet import UNet
from corrupt_data import CorruptedMNIST
import argparse
import os
from torchvision.utils import save_image, make_grid
from evaluate import evaluate

parser = argparse.ArgumentParser(description="Train baseline model on CorruptedMNIST")
parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs to train for")
parser.add_argument("--device", type=str, default=None, help="Device to run on ('cuda', 'cpu'); if None, auto-detect")
parser.add_argument("--output_folder", type=str, default="outputs", help="Folder to save outputs/checkpoints")
parser.add_argument("--eval_every", type=int, default=1000, help="Run evaluation every N iterations (not used in baseline)")
parser.add_argument("--observed_mask_pct", type=float, default=0.1, help="Percentage of pixels that are observed (not masked) in the dataset")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")

args = parser.parse_args()

n_epochs = getattr(args, "n_epochs", 10)
output_folder = getattr(args, "output_folder", "outputs")
eval_every = getattr(args, "eval_every", 1000)
batch_size = getattr(args, "batch_size", 64)
device = getattr(args, "device", 'cuda' if torch.cuda.is_available() else 'cpu')
OBSERVED_MASK_PCT = getattr(args, "observed_mask_pct", 0.1)
if OBSERVED_MASK_PCT > 1:
    OBSERVED_MASK_PCT = OBSERVED_MASK_PCT / 100

print("Arguments:")
print(f"n_epochs: {n_epochs}")
print(f"output_folder: {output_folder}")
print(f"eval_every: {eval_every}")
print(f"batch_size: {batch_size}")
print(f"device: {device}")
print(f"OBSERVED_MASK_PCT: {OBSERVED_MASK_PCT}")


# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

def train():
    
    # DATASET: Use observed_mask_pct from command line
    dataset = CorruptedMNIST(mask_percentage=OBSERVED_MASK_PCT, train=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    model = UNet().to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of model parameters: {num_params}")
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Track total iterations for evaluation
    iteration = 0
    
    for epoch in range(n_epochs):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        
        for x_obs, mask_obs, _ in pbar:
            iteration += 1
            # x_obs: -1 (masked) or 0/1 (visible)
            x_obs = x_obs.to(device)
            mask_obs = mask_obs.to(device)
            b = x_obs.shape[0]
            
            t_easy = torch.rand(b, device=device) * (1.0 - OBSERVED_MASK_PCT) + OBSERVED_MASK_PCT
            
            # Create extra mask
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

            print("Shapes: ", x_input.shape, pred_logits.shape)
            
            # Loss Calculation
            # We only learn on pixels that are:
            # 1. Currently Masked (final_mask == 1)
            # 2. BUT were visible in x_obs (mask_obs == 0)
            learnable_region = (final_mask == 1) & (mask_obs == 0)
            
            # Target is x_obs (0.0 or 1.0)
            print(x_obs.shape, pred_logits.shape)
            loss = F.binary_cross_entropy_with_logits(pred_logits, x_obs, reduction='none')
            print("Loss shapes: ", loss.shape, learnable_region.shape, t_easy.shape)
            loss = (loss * learnable_region * t_easy).sum() / (learnable_region.sum() + 1e-6)
            
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            
            pbar.set_postfix(loss=loss.item(), iter=iteration)

            # Evaluate every eval_every iterations
            if iteration % eval_every == 0:
                evaluate(model, device, output_folder, iteration, OBSERVED_MASK_PCT)
        # Save the model every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(output_folder, f"baseline_mnist_epoch{epoch+1}.pth"))
    torch.save(model.state_dict(), os.path.join(output_folder, "baseline_mnist_final.pth"))
    print("Training Done.")

if __name__ == "__main__":
    train()