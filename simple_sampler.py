import torch
from model import UNet
from torchvision.utils import save_image

def iterative_sampling(model, device, steps=10):
    model.eval()
    b, c, h, w = 16, 1, 28, 28
    
    # Start: Fully masked (all zeros)
    current_img = torch.zeros(b, c, h, w).to(device)
    # Mask: 0 means "unknown/masked", 1 means "known"
    # Initially all 0
    current_mask = torch.zeros(b, 1, h, w).to(device)
    
    # We want to go from 0% known to 100% known in 'steps'
    pixels_per_step = (h * w) // steps
    
    with torch.no_grad():
        for i in range(steps):
            # 1. Prepare timestep (conceptually, how much is masked?)
            # At i=0, 100% masked (t=1000). At i=steps, 0% masked (t=0).
            t_val = 1000 - (i * (1000 // steps))
            t = torch.full((b,), t_val, device=device).long()
            
            # 2. Predict x_0 from current state
            # current_img has 0s in masked spots, values in known spots
            pred_x_0 = model(current_img, t, current_mask)
            
            # 3. Select new pixels to "reveal"
            # In VQ-VAE we'd pick high probability tokens. 
            # Here, we just pick random locations that are currently masked (0).
            
            # This is a bit tricky with tensors. 
            # Simplified approach: Just take the prediction, and accept 'pixels_per_step' chunks.
            
            # Update the mask: Set 'pixels_per_step' random 0s to 1s
            # (In a real implementation we do this per image, here simplified for batch)
            
            # ACTUALLY, simpler approach for continuous MNIST:
            # Just mix the prediction into the current image based on the schedule.
            
            # Update our "Known" image with the prediction for the unknown parts
            current_img = current_mask * current_img + (1 - current_mask) * pred_x_0
            
            # Update mask for next step (reveal more)
            # Create a mask that has slightly more 1s than before
            ratio = (i + 1) / steps
            rand = torch.rand(b, 1, h, w).to(device)
            new_mask = (rand < ratio).float()
            
            # Ensure we don't "forget" pixels we already revealed (union of masks)
            current_mask = torch.max(current_mask, new_mask)
            
            # Enforce the mask on the image for the next pass
            # (Keep known pixels, zero out the rest)
            current_img = current_img * current_mask
            
    # Final cleanup: The last prediction is the result
    return pred_x_0

# Run
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = UNet(in_channels=1, out_channels=1, base_channels=64).to(device)
model.load_state_dict(torch.load("masked_diffusion.pth"))

samples = iterative_sampling(model, device)
save_image(samples, "final_generated_digits.png", normalize=True)