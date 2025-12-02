import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
import os

class ColoredMNIST(Dataset):
    """
    Colored MNIST dataset - colors MNIST digits with random hues.
    Based on the implementation from Skoltech.
    """
    def __init__(self, root, train=True, download=True, transform=None, image_size=32):
        self.mnist = datasets.MNIST(root=root, train=train, download=download, transform=None)
        self.transform = transform
        self.image_size = image_size
        self.hues = 360 * torch.rand(len(self.mnist))

    def __len__(self):
        return len(self.mnist)

    def color_image(self, img, idx):
        """Convert grayscale image to colored image"""
        img_min = 0
        a = (img - img_min) * (self.hues[idx] % 60) / 60
        img_inc = a
        img_dec = img - a

        colored_image = torch.zeros((3, img.shape[1], img.shape[2]))
        H_i = round(self.hues[idx].item() / 60) % 6

        if H_i == 0:
            colored_image[0] = img
            colored_image[1] = img_inc
            colored_image[2] = img_min
        elif H_i == 1:
            colored_image[0] = img_dec
            colored_image[1] = img
            colored_image[2] = img_min
        elif H_i == 2:
            colored_image[0] = img_min
            colored_image[1] = img
            colored_image[2] = img_inc
        elif H_i == 3:
            colored_image[0] = img_min
            colored_image[1] = img_dec
            colored_image[2] = img
        elif H_i == 4:
            colored_image[0] = img_inc
            colored_image[1] = img_min
            colored_image[2] = img
        elif H_i == 5:
            colored_image[0] = img
            colored_image[1] = img_min
            colored_image[2] = img_dec

        return colored_image

    def __getitem__(self, idx):
        img, label = self.mnist[idx]
        # Convert PIL Image to tensor if needed
        if not isinstance(img, torch.Tensor):
            import torchvision.transforms as T
            to_tensor = T.ToTensor()
            img = to_tensor(img)
        colored_img = self.color_image(img, idx)
        
        # Resize to target image size if needed
        if colored_img.shape[1] != self.image_size or colored_img.shape[2] != self.image_size:
            import torch.nn.functional as F
            colored_img = F.interpolate(
                colored_img.unsqueeze(0), 
                size=(self.image_size, self.image_size),
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
        
        # Normalize to [-1, 1]
        colored_img = colored_img * 2.0 - 1.0
        
        return colored_img, label

class ImageFolderDataset(Dataset):
    """Simple image folder dataset"""
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_paths = []
        
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            self.image_paths.extend([
                os.path.join(root, f) for f in os.listdir(root) 
                if f.lower().endswith(ext.replace('*', ''))
            ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        else:
            # Default transform
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
            img = transform(img)
        
        return img

def get_dataloader(data_dir='./data', batch_size=64, image_size=32, 
                   num_workers=4, use_colored_mnist=True):
    """
    Get dataloader for training.
    
    Args:
        data_dir: Data directory
        batch_size: Batch size
        image_size: Image size
        num_workers: Number of workers
        use_colored_mnist: Whether to use ColoredMNIST (default) or ImageFolder
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])
    
    if use_colored_mnist:
        # Use ColoredMNIST
        dataset = ColoredMNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=None,
            image_size=image_size
        )
    else:
        # Use ImageFolder
        dataset = ImageFolderDataset(root=data_dir, transform=transform)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader

