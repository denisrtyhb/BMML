import torch
from torchvision.utils import save_image
from unet import UNet

import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Sample from the Ambient Model")
    parser.add_argument("--device", default=None, type=str, help="Device to use: 'cuda' or 'cpu'")
    parser.add_argument("--model_path", type=str, help="Path to model weights")
    parser.add_argument("--output_path", default="ambient_generated.png", type=str, help="Output image path")
    return parser.parse_args()

args = parse_args()
device = getattr(args, "device", 'cuda' if torch.cuda.is_available() else 'cpu')
model_path = args.model_path
output_path = args.output_path

def simple_sample():
    model = UNet().to(device)
    
    model.load_state_dict(torch.load(model_path))
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
            
    dir = os.path.dirname(output_path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    save_image(img, output_path)
    print(f"Saved image to {output_path}")

if __name__ == "__main__":
    simple_sample()