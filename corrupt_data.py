import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

class CorruptedMNIST(Dataset):
    def __init__(self, root='./data', train=True, mask_percentage=0.5, download=True):
        """
        A wrapper around MNIST that permanently masks a percentage of pixels.
        This simulates the 'Ambient' setting where we only have access to corrupted data.
        """
        # 1. Load standard MNIST
        # We use the same binarization transform as before for sharp digits
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x > 0.5).float()) # Binarize (0 or 1)
        ])
        
        self.mnist = datasets.MNIST(root, train=train, download=download, transform=self.transform)
        self.mask_percentage = mask_percentage
        
        # 2. Pre-calculate masks (Optional)
        # In a real Ambient setting, the mask is usually fixed per image.
        # We will generate a consistent mask for every index to simulate this.
        # (We use a deterministic seed based on the index so it's reproducible)
        self.fixed_masks = {}

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        # Get clean image (0 or 1)
        clean_img, label = self.mnist[idx]
        
        # Generate a deterministic mask for this specific image index
        # (simulating that we found this photo already torn/damaged)
        if idx not in self.fixed_masks:
            g = torch.Generator()
            g.manual_seed(idx) # Deterministic mask based on image ID
            
            # Create probability matrix
            probs = torch.full_like(clean_img, self.mask_percentage)
            
            # 1 = Masked, 0 = Visible
            # We use the generator to make it deterministic per image
            mask = torch.bernoulli(probs, generator=g)
            self.fixed_masks[idx] = mask
        else:
            mask = self.fixed_masks[idx]
        
        # Apply Mask
        # Visible (mask=0) -> Keep clean_img (0 or 1)
        # Masked (mask=1)  -> Set to -1.0
        corrupted_img = clean_img * (1 - mask) - 1.0 * mask
        
        # In Ambient Diffusion, we return:
        # 1. The Corrupted Image (this is our "x_observed")
        # 2. The Mask (so we know which pixels are valid)
        # 3. The Label (optional)
        return corrupted_img, mask, label

def visualize_corruption(percentage=0.5):
    """
    Helper function to see what the dataset looks like.
    """
    print(f"Loading MNIST with {percentage*100}% corruption...")
    dataset = CorruptedMNIST(mask_percentage=percentage, train=False)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Get a batch
    corrupted_imgs, masks, _ = next(iter(loader))
    
    # Plot
    fig, axs = plt.subplots(2, 4, figsize=(12, 6))
    plt.suptitle(f"Ambient MNIST (Observed Data) - {percentage*100}% Masked", fontsize=16)
    
    for i in range(8):
        ax = axs[i//4, i%4]
        # Reshape to 28x28
        img = corrupted_imgs[i].squeeze().numpy()
        
        # -1 is Black, 0 is Gray (conceptually), 1 is White.
        # But for plotting, let's map -1 to 0.5 (Gray) to see the holes clearly,
        # or just plot as is.
        ax.imshow(img, cmap='gray', vmin=-1, vmax=1)
        ax.set_title(f"Sample {i}")
        ax.axis('off')
    
    save_path = "ambient_mnist_preview.png"
    plt.savefig(save_path)
    print(f"Saved preview to {save_path}")

if __name__ == "__main__":
    # Test with 50% missing pixels
    visualize_corruption(percentage=0.5)