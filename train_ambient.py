import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
# from unet import UNet
from unet2 import MDMUNet
from corrupt_data import CorruptedMNIST
import argparse
import os
from torchvision.utils import save_image, make_grid
from evaluate import evaluate

parser = argparse.ArgumentParser(description="Train ambient model on CorruptedMNIST")
parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs to train for")
parser.add_argument("--device", type=str, default=None, help="Device to run on ('cuda', 'cpu'); if None, auto-detect")
parser.add_argument("--output_folder", type=str, default="outputs", help="Folder to save outputs/checkpoints")
parser.add_argument("--consistency_weight", type=float, default=1.0, help="Weight for consistency loss")
parser.add_argument("--eval_every", type=int, default=1000, help="Run evaluation every N iterations")
parser.add_argument("--observed_mask_pct", type=float, default=0.1, help="Percentage of pixels that are observed (not masked) in the dataset")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
parser.add_argument("--start_consistency_epoch", type=int, default=0, help="Epoch to start consistency loss")
args = parser.parse_args()

n_epochs = getattr(args, "n_epochs", 10)
output_folder = getattr(args, "output_folder", "outputs")
consistency_weight = getattr(args, "consistency_weight", 1.0)
batch_size = getattr(args, "batch_size", 64)
device = getattr(args, "device", 'cuda' if torch.cuda.is_available() else 'cpu')
start_consistency_epoch = getattr(args, "start_consistency_epoch", 10000)


OBSERVED_MASK_PCT = getattr(args, "observed_mask_pct", 0.1)
if OBSERVED_MASK_PCT > 1:
    OBSERVED_MASK_PCT = OBSERVED_MASK_PCT / 100
print("Arguments:")
print(f"n_epochs: {n_epochs}")
print(f"output_folder: {output_folder}")
print(f"consistency_weight: {consistency_weight}")
print(f"start_consistency_epoch: {start_consistency_epoch}")
print(f"batch_size: {batch_size}")
print(f"device: {device}")
print(f"OBSERVED_MASK_PCT: {OBSERVED_MASK_PCT}")

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

def increase_mask_percentage(mask, mask_percentage):
    """
    Sets approx mask_percentage of all entries in the mask to 1, and the rest to 0.
    Picking (1-mask_percentage) of ones and flipping them to zero (i.e., increasing masking).
    Works for batched masks as well.
    mask: torch.Tensor of shape (B, 1, H, W) or (B, H, W) with values 0 or 1
    mask_percentage: float, between 0 and 1
    Returns: torch.Tensor with modified mask
    """
    mask_flat = mask.view(mask.shape[0], -1)
    device = mask.device
    batch_size, N = mask_flat.shape

    out_mask_flat = mask_flat.clone()

    for i in range(batch_size):
        ones_idx = (mask_flat[i] == 1).nonzero(as_tuple=True)[0]
        n_ones = ones_idx.numel()
        n_keep = int(round(n_ones * mask_percentage))
        if n_keep == 0:
            out_mask_flat[i, ones_idx] = 0
        else:
            # Randomly select indices to keep as 1, the rest set to 0
            perm = torch.randperm(n_ones, device=device)
            keep_idx = ones_idx[perm[:n_keep]]
            zero_idx = ones_idx[perm[n_keep:]]
            out_mask_flat[i, zero_idx] = 0
            out_mask_flat[i, keep_idx] = 1

    return out_mask_flat.view_as(mask)

def calculate_consistency_loss(model, x_obs, t_obs, device, step_back=0.1):
    """
    CONSISTENCY LOSS (The 'Hard' Part)
    Training for t < t_dataset by ensuring the model makes consistent predictions.
    """
    b, c, h, w = x_obs.shape
    
    mask_obs = (x_obs == -1).float()

    pred_logits_student = model(x_obs, t_obs * 1000, mask_obs)
    pred_x0_student_probs = torch.sigmoid(pred_logits_student)

    with torch.no_grad():
        pred_x0_hard = torch.bernoulli(pred_x0_student_probs)
        teacher_mask = increase_mask_percentage(mask_obs, step_back)

        x_teacher = x_obs * (1 - teacher_mask) + pred_x0_hard * teacher_mask
        t_teacher = torch.clamp(t_obs - step_back, min=0.001)

        pred_logits_teacher = model(x_teacher, t_teacher * 1000, teacher_mask)
        pred_x0_teacher_probs = torch.sigmoid(pred_logits_teacher)
    
    cons_loss = F.binary_cross_entropy_with_logits(pred_logits_student, pred_x0_teacher_probs)
    
    return cons_loss

from evaluate import evaluate

def train():
    # DATASET: 10% Corrupted (Ambient Setting from command line)
    dataset = CorruptedMNIST(mask_percentage=OBSERVED_MASK_PCT, train=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    model = MDMUNet().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Track total iterations for evaluation
    iteration = 0
    
    for epoch in range(n_epochs):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        
        easy_loss_total = 0
        hard_loss_total = 0
        num_batches = 0

        for x_obs, mask_obs, _ in pbar:
            iteration += 1
            # x_obs: -1 (masked) or 0/1 (visible)
            x_obs = x_obs.to(device)
            mask_obs = mask_obs.to(device)
            b = x_obs.shape[0]
            
            # --- TASK 1: EASY LOSS (Ambient Denoising) ---
            # Train on noise levels HIGHER than dataset (e.g. 0.6, 0.8)
            # We take visible pixels and hide MORE of them.
            
            # Sample t between 0.5 and 1.0
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
            
            # Loss Calculation
            # We only learn on pixels that are:
            # 1. Currently Masked (final_mask == 1)
            # 2. BUT were visible in x_obs (mask_obs == 0)
            learnable_region = (final_mask == 1) & (mask_obs == 0)
            
            # Target is x_obs (0.0 or 1.0)
            loss_easy = F.binary_cross_entropy_with_logits(pred_logits, x_obs, reduction='none')
            loss_easy = (loss_easy * learnable_region).sum() / (learnable_region.sum() + 1e-6)
            
            # --- TASK 2: HARD LOSS (Consistency) ---
            # Train on noise levels LOWER than dataset (e.g. 0.4, 0.2)

            if epoch < start_consistency_epoch:
                loss_hard = 0
            else:
                t_dataset = torch.full((b,), OBSERVED_MASK_PCT, device=device)
                loss_hard = calculate_consistency_loss(model, x_obs, t_dataset, device)
    
            loss = loss_easy + loss_hard * consistency_weight
            
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            
            easy_loss_total += loss_easy.item()
            hard_loss_total += loss_hard.item()
            num_batches += 1

            if epoch < start_consistency_epoch:
                pbar.set_postfix(easy=easy_loss_total/num_batches, iter=iteration)
            else:
                pbar.set_postfix(easy=easy_loss_total/num_batches, hard=hard_loss_total/num_batches, iter=iteration)
            
            # Evaluate every eval_every iterations
        evaluate(model, device, output_folder, iteration, OBSERVED_MASK_PCT)
        
        # Save the model every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(output_folder, f"ambient_mnist_epoch{epoch+1}.pth"))
    torch.save(model.state_dict(), os.path.join(output_folder, "ambient_mnist_final.pth"))
    print("Training Done.")

if __name__ == "__main__":
    train()