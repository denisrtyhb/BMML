import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import os

# Import your modules
from unet import UNet
from corrupt_data import CorruptedMNIST

def sample_intermediate(model, device, epoch, save_dir="training_logs"):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    steps = 40
    b = 16
    img = torch.full((b, 1, 28, 28), -1.0, device=device)
    mask = torch.ones(b, 1, 28, 28).to(device)
    
    with torch.no_grad():
        for i in range(steps):
            t_vec = torch.full((b,), 1.0 - (i / steps), device=device)
            pred_logits = model(img, t_vec * 1000, mask)
            probs = torch.sigmoid(pred_logits)
            sampled_prediction = torch.bernoulli(probs)
            img = (1 - mask) * img + mask * sampled_prediction
            next_t = 1.0 - ((i + 1) / steps)
            new_mask = (torch.rand(b, 1, 28, 28).to(device) < next_t).float()
            mask = mask * new_desired_mask = new_mask # Logic fix: intersection
            mask = torch.max(torch.zeros_like(mask), new_mask) # Simplified random walk for viz
            # Actually for viz just random shrinking is fine
            mask = (torch.rand(b, 1, 28, 28).to(device) < next_t).float()
            img = img * (1 - mask) - 1.0 * mask
            
    save_image(img, os.path.join(save_dir, f"epoch_{epoch}.png"), nrow=4)

def calculate_unbiased_consistency_loss(model, x_obs, t_obs, device):
    """
    STRICT IMPLEMENTATION OF EQ 3.6 / VARIANCE REDUCTION
    1. Teacher predicts x0.
    2. Generate Student 1 (Mask A).
    3. Generate Student 2 (Mask B).
    4. Loss = (Student1 - Teacher) * (Student2 - Teacher)
    """
    b, c, h, w = x_obs.shape
    
    # --- 1. TEACHER STEP ---
    with torch.no_grad():
        mask_obs = (x_obs == -1).float()
        # Get Teacher Logits
        logits_teacher = model(x_obs, t_obs * 1000, mask_obs)
        # Convert to Probability (This is h_theta in the paper)
        h_teacher = torch.sigmoid(logits_teacher)
        
        # Create the "Hallucinated" Clean Image to generate students from
        # We sample a hard 0/1 x0 from the teacher's distribution
        x0_hard_sample = torch.bernoulli(h_teacher)
        
        # Fill the holes in observation
        x_filled = x_obs * (1 - mask_obs) + x0_hard_sample * mask_obs

    # --- SETUP FOR STUDENTS ---
    # Target time for students (slightly less noise)
    t_student = torch.clamp(t_obs - 0.1, min=0.001)
    mask_probs = t_student.view(-1, 1, 1, 1).expand(b, 1, h, w)

    # --- 2. STUDENT 1 ---
    # Generate random mask A
    mask_1 = (torch.rand(b, 1, h, w, device=device) < mask_probs).float()
    # Apply mask to filled image
    x_student_1 = x_filled * (1 - mask_1) - 1.0 * mask_1
    # Predict
    logits_1 = model(x_student_1, t_student * 1000, mask_1)
    h_student_1 = torch.sigmoid(logits_1)

    # --- 3. STUDENT 2 ---
    # Generate random mask B (Independent from A)
    mask_2 = (torch.rand(b, 1, h, w, device=device) < mask_probs).float()
    # Apply mask to filled image
    x_student_2 = x_filled * (1 - mask_2) - 1.0 * mask_2
    # Predict
    logits_2 = model(x_student_2, t_student * 1000, mask_2)
    h_student_2 = torch.sigmoid(logits_2)

    # --- 4. CALCULATE LOSS (The Paper's Equation) ---
    # Eq: (h_s1 - h_t)^T * (h_s2 - h_t)
    
    # Difference vectors
    diff_1 = h_student_1 - h_teacher
    diff_2 = h_student_2 - h_teacher
    
    # Dot Product (Sum over pixels C, H, W)
    # We keep Batch dimension to average later
    # flatten start_dim=1 means [B, C*H*W]
    dot_product = (diff_1.flatten(1) * diff_2.flatten(1)).sum(dim=1)
    
    # Final Loss (Average over batch)
    loss = dot_product.mean()
    
    return loss

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    OBSERVED_MASK_PCT = 0.5
    dataset = CorruptedMNIST(mask_percentage=OBSERVED_MASK_PCT, train=True)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)
    
    model = UNet().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    epochs = 20
    
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        
        for x_obs, mask_obs, _ in pbar:
            x_obs = x_obs.to(device)
            mask_obs = mask_obs.to(device)
            b = x_obs.shape[0]
            
            # --- TASK 1: EASY LOSS (Standard Denoising) ---
            t_easy = torch.rand(b, device=device) * (1.0 - OBSERVED_MASK_PCT) + OBSERVED_MASK_PCT
            mask_probs = t_easy.view(-1, 1, 1, 1).expand(b, 1, 28, 28)
            extra_mask = (torch.rand_like(x_obs) < mask_probs).float()
            final_mask = torch.max(mask_obs, extra_mask)
            
            x_input = x_obs.clone()
            x_input[extra_mask == 1] = -1.0
            
            pred_logits = model(x_input, t_easy * 1000, final_mask)
            
            learnable_region = (final_mask == 1) & (mask_obs == 0)
            # Standard BCE for the easy part (reconstruction)
            loss_easy = F.binary_cross_entropy_with_logits(pred_logits, x_obs, reduction='none')
            loss_easy = (loss_easy * learnable_region).sum() / (learnable_region.sum() + 1e-6)
            
            # --- TASK 2: HARD LOSS (Unbiased Consistency) ---
            t_dataset = torch.full((b,), OBSERVED_MASK_PCT, device=device)
            
            # Using the strict "Two Students" implementation
            loss_hard = calculate_unbiased_consistency_loss(model, x_obs, t_dataset, device)
            
            # Total Loss
            loss = loss_easy + loss_hard
            
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            
            pbar.set_postfix(easy=loss_easy.item(), hard=loss_hard.item())
            
        if (epoch + 1) % 5 == 0:
            sample_intermediate(model, device, epoch+1)
            model.train()
            
    torch.save(model.state_dict(), "mdlm_unbiased_consistency.pth")
    print("Training Done.")

if __name__ == "__main__":
    train()
