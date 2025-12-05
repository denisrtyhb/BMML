import torch
from torchvision.utils import save_image
from unet import UNet

# def simple_sample():
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     model = UNet().to(device)
#     model.load_state_dict(torch.load("mdlm_mnist.pth"))
#     model.eval()
    
#     steps = 20
#     b = 16
    
#     # Start: Fully Masked Image (All Zeros)
#     img = torch.zeros(b, 1, 28, 28).to(device)
    
#     # Start: Mask is all 1s (Everything is masked)
#     mask = torch.ones(b, 1, 28, 28).to(device)
    
#     print("Sampling...")
#     with torch.no_grad():
#         for i in range(steps):
#             # Current time t goes from 1.0 down to 0.0
#             t_val = 1.0 - (i / steps)
#             t_vec = torch.full((b,), t_val, device=device)
            
#             # 1. Predict the clean image x0 from current state
#             # Pass t*1000 for embedding
#             pred_x0 = model(img, t_vec * 1000, mask)
            
#             # 2. Update Image
#             # For pixels that are currently masked (mask=1), use prediction.
#             # For pixels that are already known (mask=0), keep them.
#             img = (1 - mask) * img + mask * pred_x0
            
#             # 3. Determine Next Mask
#             # We want to have fewer masked pixels in next step.
#             # Next t will be:
#             next_t_val = 1.0 - ((i + 1) / steps)
            
#             # We keep pixels based on probability.
#             # If a pixel is masked, probability it STAYS masked is ratio of t's.
#             # But simpler "Greedy" approach (like MaskGIT):
#             # Just keep the top confident ones?
#             # Or simpler random approach (MDLM standard):
            
#             # Create a new random mask where prob(1) = next_t_val
#             # BUT we must respect the previous mask (cannot re-mask a known pixel).
            
#             random_mask = torch.rand(b, 1, 28, 28).to(device)
#             new_desired_mask = (random_mask < next_t_val).float()
            
#             # The actual mask is the intersection: 
#             # Has to be in new_desired_mask AND was previously masked.
#             # (Ideally in MDLM we re-sample, but for MNIST shrinking mask is fine)
#             mask = mask * new_desired_mask
            
#             # Apply mask to image (zero out the parts that are still masked)
#             img = img * (1 - mask) # + 0 * mask
            
#     # Final save
#     save_image(img, "mdlm_generated_1.png", normalize=True)
#     print("Saved mdlm_generated.png")

# if __name__ == "__main__":
#     simple_sample()
# import torch
# from torchvision.utils import save_image
# from unet import UNet

# def simple_sample():
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     model = UNet().to(device)
    
#     # Load the trained model
#     try:
#         model.load_state_dict(torch.load("mdlm_mnist.pth"))
#     except FileNotFoundError:
#         # Try full path if running from root
#         model.load_state_dict(torch.load("/kaggle/working/simple_gpt_try/mdlm_mnist.pth"))
        
#     model.eval()
    
#     steps = 40  # More steps = better quality
#     b = 16
    
#     # --- FIX 1: Start with -1.0 (Black), not 0 (Gray) ---
#     img = torch.full((b, 1, 28, 28), -1.0, device=device)
    
#     # Start: Mask is all 1s (Everything is masked)
#     mask = torch.ones(b, 1, 28, 28).to(device)
    
#     print("Sampling...")
#     with torch.no_grad():
#         for i in range(steps):
#             # t goes from 1.0 down to 0.0
#             t_val = 1.0 - (i / steps)
#             t_vec = torch.full((b,), t_val, device=device)
            
#             # 1. Predict
#             # Model inputs: x (with -1s), t, mask
#             pred_x0 = model(img, t_vec * 1000, mask)
            
#             # 2. Update Image
#             # If mask is 0 (known), keep img. If mask is 1 (unknown), use pred.
#             img = (1 - mask) * img + mask * pred_x0
            
#             # 3. Determine Next Mask
#             next_t_val = 1.0 - ((i + 1) / steps)
            
#             # Generate random matrix to decide what to keep
#             random_mask = torch.rand(b, 1, 28, 28).to(device)
#             new_desired_mask = (random_mask < next_t_val).float()
            
#             # Intersection: Can only mask if it was ALREADY masked
#             mask = mask * new_desired_mask
            
#             # --- FIX 2: Re-apply -1.0 to the remaining masked regions ---
#             # If we just multiply by (1-mask), it becomes 0 (Gray).
#             # We must subtract mask to make it -1 (Black).
#             # (Assuming img is normalized [-1, 1])
            
#             # Logic:
#             # Visible (mask=0) -> keep image value
#             # Masked (mask=1) -> set to -1.0
#             img = img * (1 - mask) - 1.0 * mask
            
#     # Final save
#     save_image(img, "mdlm_generated_fixed.png", normalize=True)
#     print("Saved mdlm_generated_fixed.png")

# if __name__ == "__main__":
#     simple_sample()
# import torch
# from torchvision.utils import save_image
# from unet import UNet

# def simple_sample():
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     model = UNet().to(device)
    
#     try:
#         model.load_state_dict(torch.load("mdlm_mnist_binary.pth"))
#     except:
#         model.load_state_dict(torch.load("/kaggle/working/simple_gpt_try/mdlm_mnist_binary.pth"))
        
#     model.eval()
    
#     steps = 40
#     b = 16
    
#     # Start: Image is 0.5 (Gray/Neutral) or 0.
#     # Since we have a mask channel, setting image to 0 is fine.
#     img = torch.zeros(b, 1, 28, 28).to(device)
    
#     # Start: Mask is all 1s
#     mask = torch.ones(b, 1, 28, 28).to(device)
    
#     print("Sampling Binary Digits...")
#     with torch.no_grad():
#         for i in range(steps):
#             t_val = 1.0 - (i / steps)
#             t_vec = torch.full((b,), t_val, device=device)
            
#             # 1. Predict Logits
#             pred_logits = model(img, t_vec * 1000, mask)
            
#             # 2. Convert to Probability (0.0 to 1.0)
#             probs = torch.sigmoid(pred_logits)
            
#             # 3. SAMPLE (The Key Step)
#             # Instead of keeping the float value, we flip a coin based on probability.
#             # This makes the image Binary (0 or 1), removing blur.
#             sampled_prediction = torch.bernoulli(probs)
            
#             # 4. Update Image
#             # Keep known pixels, fill unknown with SAMPLED prediction
#             img = (1 - mask) * img + mask * sampled_prediction
            
#             # 5. Determine Next Mask
#             next_t_val = 1.0 - ((i + 1) / steps)
#             random_mask = torch.rand(b, 1, 28, 28).to(device)
#             new_desired_mask = (random_mask < next_t_val).float()
            
#             # Intersection
#             mask = mask * new_desired_mask
            
#             # Apply mask to image (Zero out unknown parts for the next pass)
#             img = img * (1 - mask)
            
#     save_image(img, "mdlm_binary_result.png")
#     print("Saved mdlm_binary_result.png")

# if __name__ == "__main__":
#     simple_sample()

import torch
from torchvision.utils import save_image
from unet import UNet

def simple_sample():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UNet().to(device)
    
    # Load the AMBIENT trained model
    try:
        model.load_state_dict(torch.load("mdlm_ambient_consistency.pth"))
    except:
        model.load_state_dict(torch.load("/kaggle/working/simple_gpt_try/mdlm_ambient_consistency.pth"))
        
    model.eval()
    
    steps = 40
    b = 16
    
    # Start: Fully Masked (-1.0)
    img = torch.full((b, 1, 28, 28), -1.0, device=device)
    mask = torch.ones(b, 1, 28, 28).to(device)
    
    print("Sampling from Ambient Model...")
    with torch.no_grad():
        for i in range(steps):
            t_val = 1.0 - (i / steps)
            t_vec = torch.full((b,), t_val, device=device)
            
            # Predict
            pred_logits = model(img, t_vec * 1000, mask)
            probs = torch.sigmoid(pred_logits)
            sampled_prediction = torch.bernoulli(probs)
            
            # Update Image
            img = (1 - mask) * img + mask * sampled_prediction
            
            # Next Mask
            next_t_val = 1.0 - ((i + 1) / steps)
            random_mask = torch.rand(b, 1, 28, 28).to(device)
            new_desired_mask = (random_mask < next_t_val).float()
            mask = mask * new_desired_mask
            
            # Re-apply -1 to masks
            img = img * (1 - mask) - 1.0 * mask
            
    save_image(img, "ambient_generated.png")
    print("Saved ambient_generated.png")

if __name__ == "__main__":
    simple_sample()