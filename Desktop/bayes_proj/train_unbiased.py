import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import os

from unet import UNet
from corrupt_data import CorruptedMNIST

# --- Weight Initialization (Crucial for Convergence) ---
def weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

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
            
            # Next mask logic
            next_t = 1.0 - ((i + 1) / steps)
            new_mask = (torch.rand(b, 1, 28, 28).to(device) < next_t).float()
            mask = mask * new_mask 
            img = img * (1 - mask) - 1.0 * mask
            
    save_image(img, os.path.join(save_dir, f"epoch_{epoch}.png"), nrow=4)

def calculate_unbiased_consistency_loss(model, x_obs, t_obs, device):
    b, c, h, w = x_obs.shape
    with torch.no_grad():
        mask_obs = (x_obs == -1).float()
        logits_teacher = model(x_obs, t_obs * 1000, mask_obs)
        h_teacher = torch.sigmoid(logits_teacher)
        x0_hard_sample = torch.bernoulli(h_teacher)
        x_filled = x_obs * (1 - mask_obs) + x0_hard_sample * mask_obs

    t_student = torch.clamp(t_obs - 0.1, min=0.001)
    mask_probs = t_student.view(-1, 1, 1, 1).expand(b, 1, h, w)

    # Student 1
    mask_1 = (torch.rand(b, 1, h, w, device=device) < mask_probs).float()
    x_student_1 = x_filled * (1 - mask_1) - 1.0 * mask_1
    logits_1 = model(x_student_1, t_student * 1000, mask_1)
    h_student_1 = torch.sigmoid(logits_1)

    # Student 2
    mask_2 = (torch.rand(b, 1, h, w, device=device) < mask_probs).float()
    x_student_2 = x_filled * (1 - mask_2) - 1.0 * mask_2
    logits_2 = model(x_student_2, t_student * 1000, mask_2)
    h_student_2 = torch.sigmoid(logits_2)

    diff_1 = h_student_1 - h_teacher
    diff_2 = h_student_2 - h_teacher
    dot_product = (diff_1.flatten(1) * diff_2.flatten(1)).sum(dim=1)
    loss = dot_product.mean()
    return loss

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    OBSERVED_MASK_PCT = 0.1
    dataset = CorruptedMNIST(mask_percentage=OBSERVED_MASK_PCT, train=True)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)
    
    model = UNet().to(device)
    model.apply(weights_init)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    epochs = 10
    
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        
        for x_obs, mask_obs, _ in pbar:
            x_obs = x_obs.to(device)
            mask_obs = mask_obs.to(device)
            b = x_obs.shape[0]
            
            # --- TASK 1: EASY LOSS ---
            t_easy = torch.rand(b, device=device) * (1.0 - OBSERVED_MASK_PCT) + OBSERVED_MASK_PCT
            mask_probs = t_easy.view(-1, 1, 1, 1).expand(b, 1, 28, 28)
            extra_mask = (torch.rand_like(x_obs) < mask_probs).float()
            final_mask = torch.max(mask_obs, extra_mask)
            
            x_input = x_obs.clone()
            x_input[extra_mask == 1] = -1.0
            
            pred_logits = model(x_input, t_easy * 1000, final_mask)
            
            learnable_region = (final_mask == 1) & (mask_obs == 0)
            
            # --- FIX STARTS HERE ---
            # 1. Clamp target to [0, 1] so BCE doesn't see -1.0
            safe_target = x_obs.clamp(0, 1) 
            
            # 2. Calculate Loss on Safe Target
            loss_easy_raw = F.binary_cross_entropy_with_logits(pred_logits, safe_target, reduction='none')
            
            # 3. Apply Mask (Only count regions that were visible in x_obs but hidden in x_input)
            loss_easy = (loss_easy_raw * learnable_region).sum() / (learnable_region.sum() + 1e-6)
            # --- FIX ENDS HERE ---
            
            # --- TASK 2: HARD LOSS ---
            t_dataset = torch.full((b,), OBSERVED_MASK_PCT, device=device)
            loss_hard = calculate_unbiased_consistency_loss(model, x_obs, t_dataset, device)
            
            # Total Loss
            loss = loss_easy + loss_hard
            
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            
            pbar.set_postfix(easy=loss_easy.item(), hard=loss_hard.item())
            
        if (epoch + 1) % 1== 0:
            sample_intermediate(model, device, epoch+1)
            model.train()
            
    torch.save(model.state_dict(), "mdlm_unbiased_consistency.pth")
    print("Training Done.")

if __name__ == "__main__":
    train()