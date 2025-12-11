import torch
from torchvision.utils import save_image
from unet2 import MDMUNet

import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Sample from the Ambient Model")
    parser.add_argument("--device", default=None, type=str, help="Device to use: 'cuda' or 'cpu'")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model weights")
    parser.add_argument("--output_path", default="ambient_generated.png", type=str, help="Output image path")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of samples to generate")
    parser.add_argument("--steps", type=int, default=40, help="Number of sampling steps")
    return parser.parse_args()

args = parse_args()
device = getattr(args, "device", 'cuda' if torch.cuda.is_available() else 'cpu')
model_path = args.model_path
output_path = args.output_path
num_samples = args.num_samples
steps = args.steps

def simple_sample():
    # Initialize model
    model = MDMUNet().to(device)
    
    # Load checkpoint - handle both regular and DataParallel saved models
    checkpoint = torch.load(model_path, map_location=device)
    
    # Check if checkpoint is a dict with 'model_state_dict' or just state_dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present (from DataParallel)
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.eval() 
    
    # Start: Fully Masked (-1.0)
    img = torch.full((num_samples, 1, 28, 28), -1.0, device=device)
    mask = torch.ones(num_samples, 1, 28, 28).to(device)
    
    print(f"Sampling {num_samples} images with {steps} steps...")
    with torch.no_grad():
        for i in range(steps):
            t_val = 1.0 - (i / steps)
            t_vec = torch.full((num_samples,), t_val, device=device)
            
            # Predict
            pred_logits = model(img, t_vec * 1000, mask)
            probs = torch.sigmoid(pred_logits)
            sampled_prediction = torch.bernoulli(probs)
            
            # Update Image
            img = (1 - mask) * img + mask * sampled_prediction
            
            # Next Mask
            next_t_val = 1.0 - ((i + 1) / steps)
            random_mask = torch.rand(num_samples, 1, 28, 28).to(device)
            new_desired_mask = (random_mask < next_t_val).float()
            mask = mask * new_desired_mask
            
            # Re-apply -1 to masks
            img = img * (1 - mask) - 1.0 * mask
    
    # Ensure output directory exists
    dir = os.path.dirname(output_path)
    if dir and not os.path.exists(dir):
        os.makedirs(dir)
    
    save_image(img, output_path)
    print(f"Saved {num_samples} samples to {output_path}")

if __name__ == "__main__":
    simple_sample()