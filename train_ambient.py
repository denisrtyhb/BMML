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

parser = argparse.ArgumentParser(description="Train ambient model on CorruptedMNIST")
parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs to train for")
parser.add_argument("--device", type=str, default=None, help="Device to run on ('cuda', 'cpu'); if None, auto-detect")
parser.add_argument("--output_folder", type=str, default="outputs", help="Folder to save outputs/checkpoints")
parser.add_argument("--consistency_weight", type=float, default=1.0, help="Weight for consistency loss")
parser.add_argument("--eval_every", type=int, default=1000, help="Run evaluation every N iterations")
parser.add_argument("--observed_mask_pct", type=float, default=0.1, help="Percentage of pixels that are observed (not masked) in the dataset")
args = parser.parse_args()

n_epochs = getattr(args, "n_epochs", 10)
output_folder = getattr(args, "output_folder", "outputs")
consistency_weight = getattr(args, "consistency_weight", 1.0)
eval_every = getattr(args, "eval_every", 1000)
device = getattr(args, "device", 'cuda' if torch.cuda.is_available() else 'cpu')
OBSERVED_MASK_PCT = getattr(args, "observed_mask_pct", 0.1)
if OBSERVED_MASK_PCT > 1:
    OBSERVED_MASK_PCT = OBSERVED_MASK_PCT / 100
print(f"Device: {device}")

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

def calculate_consistency_loss(model, x_obs, t_obs, device):
    """
    CONSISTENCY LOSS (The 'Hard' Part)
    Training for t < t_dataset by ensuring the model makes consistent predictions.
    """
    b, c, h, w = x_obs.shape
    
    # --- TEACHER STEP ---
    # 1. Predict full clean image from the corrupted observation
    with torch.no_grad():
        mask_obs = (x_obs == -1).float()
        pred_logits_teacher = model(x_obs, t_obs * 1000, mask_obs)
        pred_x0_teacher_probs = torch.sigmoid(pred_logits_teacher)
        
        # 2. Impute: Fill the holes in x_obs with the teacher's guess
        # We sample a hard 0/1 to simulate a valid clean image
        pred_x0_hard = torch.bernoulli(pred_x0_teacher_probs)
        
        # Construct the "Hallucinated Clean Image"
        # Keep real pixels where we have them, use prediction where we don't
        x_filled = x_obs * (1 - mask_obs) + pred_x0_hard * mask_obs

    # --- STUDENT STEP ---
    # 3. Create a state with LESS noise than observation
    # e.g. if obs is 0.1 masked, student sees 0.05 masked
    t_student = t_obs - 0.05
    t_student = torch.clamp(t_student, min=0.001)
    
    # Generate new mask for student
    mask_probs = t_student.view(-1, 1, 1, 1).expand(b, 1, h, w)
    student_mask = (torch.rand(b, 1, h, w, device=device) < mask_probs).float()
    
    # Create Student Input
    # Apply student_mask to the x_filled image
    # Visible -> Keep x_filled (0 or 1)
    # Masked -> Set to -1
    x_student = x_filled * (1 - student_mask) - 1.0 * student_mask
    
    # 4. Predict
    student_logits = model(x_student, t_student * 1000, student_mask)
    
    # --- LOSS ---
    # The student should produce the same probabilities as the teacher
    # (Soft Targets)
    cons_loss = F.binary_cross_entropy_with_logits(student_logits, pred_x0_teacher_probs)
    
    return cons_loss

from evaluate import evaluate

def train():
    # DATASET: 10% Corrupted (Ambient Setting)
    OBSERVED_MASK_PCT = 0.1
    dataset = CorruptedMNIST(mask_percentage=OBSERVED_MASK_PCT, train=True)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)
    
    model = UNet().to(device)
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
            t_dataset = torch.full((b,), OBSERVED_MASK_PCT, device=device)
            loss_hard = calculate_consistency_loss(model, x_obs, t_dataset, device)
            
            # Total Loss
    
            loss = loss_easy + loss_hard * consistency_weight
            
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            
            pbar.set_postfix(easy=loss_easy.item(), hard=loss_hard.item(), iter=iteration)
            
            # Evaluate every eval_every iterations
            if iteration % eval_every == 0:
                evaluate(model, device, output_folder, iteration, OBSERVED_MASK_PCT)
        
        # Save the model every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(output_folder, f"ambient_mnist_epoch{epoch+1}.pth"))
    torch.save(model.state_dict(), os.path.join(output_folder, "ambient_mnist_final.pth"))
    print("Training Done.")

if __name__ == "__main__":
    train()