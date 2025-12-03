import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from model import UNet
import sys

MODEL_PATH = sys.argv[1]
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
seed = 42
torch.manual_seed(seed)

def forward_process(x_0, t, mask):
    """Apply masking: x_t = mask * x_0"""
    return mask * x_0

def denoise(model, x_t, t, mask, device):
    """Denoise by predicting original image"""
    model.eval()
    with torch.no_grad():
        pred_x_0 = model(x_t, t, mask)
    return pred_x_0


import numpy as np
from PIL import Image
def save_as_image(tensor, path):
    tensor = tensor.clamp(-1, 1)
    tensor = (tensor + 1) / 2.0
    tensor = tensor.cpu().permute(1, 2, 0).numpy()
    tensor = tensor.squeeze()
    tensor = tensor * 255.0
    tensor = tensor.astype(np.uint8)
    image = Image.fromarray(tensor)
    image.save(path)

def denoise_one_step_based_on_mask(model, before, mask, t, num_to_unmask=None):
    assert before.shape == (1, 28, 28), f"Before shape: {before.shape}"
    assert mask.shape == (1, 28, 28), f"Mask shape: {mask.shape}"
    
    pred = denoise(model, before.unsqueeze(0), t.unsqueeze(0), mask.unsqueeze(0), device)[0]

    save_as_image(pred, f"preds/{(mask == 1).sum().item()}.png")

    new_pred = pred[mask == 1]
    after = before.clone()
    
    if num_to_unmask is not None:
        masked_candidates = (1-mask).nonzero(as_tuple=False)
        print(f"Unmask {num_to_unmask} pixels out of {len(masked_candidates)}")
        unmask_indices = masked_candidates[torch.randperm(len(masked_candidates))[:num_to_unmask]]

        x = unmask_indices[:, 0]
        y = unmask_indices[:, 1]
        z = unmask_indices[:, 2]

        after[x, y, z] = pred[x, y, z].clamp(-1, 1)
        mask[x, y, z] = 1
    else:
        print(f"{(mask > 0).sum()=}")
        after[mask == 1] = pred[mask == 1].clamp(-1, 1)
        mask[mask == 1] = 1

        # return pred * mask + before * (1 - mask), mask

    return after, mask

def denoise_full_trajectory_based_on_mask(model, before, mask, t, num_steps):
    # return denoise_one_step_based_on_mask(model, before, mask, t, num_to_unmask=int((1-mask).sum().item()))[0].unsqueeze(0)
    print(f"{before.shape=} {mask.shape=} {t.shape=}")
    trajectory = []
    print("New trajectory")
    for i in range(num_steps-1, -1, -1):
        num_to_unmask = int((1-mask).sum().item() // (i + 1))
        print(f"{num_to_unmask=}")
        before, mask = denoise_one_step_based_on_mask(
            model, before, mask,
            t * (i + 1) / num_steps, num_to_unmask=num_to_unmask)
        trajectory.append(before.clone())
    return torch.stack(trajectory, dim=0)

def sanity_check_denosing(num_samples=16):
    # Load model
    model = UNet.load_checkpoint(MODEL_PATH).to(device)
    # Load dataset
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=num_samples, shuffle=True)
    
    # Get samples
    images, _ = next(iter(dataloader))
    images = images.to(device)  # [B, 1, 28, 28]
    # Adding just a bit of noise
    images = images + torch.randn_like(images) * 0.1
    print("LUL")

    print("Stats for images: ", images.min(), images.max(), images.mean(), images.std())
    
    # Apply masking (like in training)
    t = torch.randint(1, 1000, (num_samples,), device=device)  # Random timestep for each sample
    prob_zero = t.float().view(-1, 1, 1, 1) / 1000.0
    random_tensor = torch.rand(num_samples, 1, 28, 28, device=device)
    mask = (random_tensor > prob_zero).float()

    print("Mask density: ", mask.mean(axis=[1,2,3]))
    
    # Forward process: create masked images
    masked = forward_process(images, t, mask)
    # Denoise
    use_denoise_full_trajectory = True
    interface_check = True
    if use_denoise_full_trajectory:
        print(mask[0].shape)
        denoised = [
            denoise_full_trajectory_based_on_mask(model, masked[i], mask[i], t[i], num_steps=2)
                for i in range(num_samples)]
        print(type(denoised))
        print([type(i) for i in denoised])
        denoised = torch.stack(denoised, dim=0)[:, -1, :, :, :]
    elif interface_check:
        print(mask[0].shape)
        denoised = [
            denoise_one_step_based_on_mask(model, masked[i], mask[i], t[i], num_to_unmask=int((1-mask[i]).sum().item()))[0]
                for i in range(num_samples)]
        print(type(denoised))
        print([type(i) for i in denoised])
        denoised = torch.stack(denoised, dim=0)

    else:
        denoised = denoise(model, masked, t, mask, device)
    
    # Prepare for visualization (denormalize to [0, 1])
    images_viz = torch.clamp((images + 1) / 2.0, 0, 1)
    masked_viz = torch.clamp((masked + 1) / 2.0, 0, 1)
    denoised_viz = torch.clamp((denoised + 1) / 2.0, 0, 1)
    
    # Create grid: original | masked | denoised
    grid = torch.cat([images_viz, masked_viz, denoised_viz], dim=0)
    grid = make_grid(grid, nrow=num_samples, padding=2)
    
    # Plot and save
    # Add left-side subtitles for each row: Original, Masked, Denoised
    for i, label in enumerate(["Original", "Masked", "Denoised"]):
        # The y-coordinate is centered for each row block
        y = (i + 0.5) / 3
        plt.figtext(0.04, 1 - y, label, va="center", ha="right", fontsize=16, weight='bold', rotation=90)
    plt.figure(figsize=(15, 5))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap='gray')
    plt.axis('off')
    plt.title('Original | Masked | Denoised')
    plt.tight_layout()
    plt.savefig('sanity_check_denosing.png', dpi=150, bbox_inches='tight')
    print(f"Saved sanity check to sanity_check_denosing.png")

def sample_images(model, num_samples=1, num_steps=10):
    # Instead of just returning the final sample, collect the images at each step and return all of them as a tensor of shape [num_steps+1, num_samples, 1, 28, 28]
    trajectory = []
    for sample in range(num_samples):
        
        before = torch.zeros(1, 28, 28, device=device)
        mask = torch.zeros(1, 28, 28, device=device)
        t = torch.tensor(999, device=device, dtype=torch.long)

        trajectory.append(denoise_full_trajectory_based_on_mask(
            model, before, mask, t, num_steps=num_steps))
    trajectory = torch.stack(trajectory, dim=0)
    return trajectory.permute(1, 0, 2, 3, 4)

def sanity_check_sampling(num_samples=3, num_steps=10):
    
    model = UNet.load_checkpoint(MODEL_PATH).to(device)
    model.eval()
    
    samples = sample_images(model, num_samples=num_samples, num_steps=num_steps)  # [num_steps+1, num_samples, 1, 28, 28]
    samples = torch.clamp(samples, -1, 1)
    samples_viz = (samples + 1) / 2.0  # [num_steps+1, num_samples, 1, 28, 28]

    # Make a grid such that each row is a trajectory over time for a single sample
    # Rearrange: [num_steps+1, num_samples, ...] -> [num_samples, num_steps+1, ...]
    samples_viz = samples_viz.permute(1, 0, 2, 3, 4)  # [num_samples, num_steps+1, 1, 28, 28]
    # Flatten trajectories: for each sample, concatenate each step along width
    samples_viz = samples_viz.reshape(num_samples * (num_steps), 1, 28, 28)
    # Make grid: num_samples rows, num_steps+1 columns
    grid = make_grid(samples_viz, nrow=num_steps, padding=2)
    plt.figure(figsize=(2 * (num_steps + 1), 2 * num_samples))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap='gray')
    plt.axis('off')
    plt.title('Sampling Trajectories (Each Row: Time Steps for One Sample)')
    plt.tight_layout()
    plt.savefig('sanity_check_sampling.png', dpi=150, bbox_inches='tight')
    print(f"Saved sanity check to sanity_check_sampling.png")

if __name__ == '__main__':
    sanity_check_denosing()
    sanity_check_sampling()
