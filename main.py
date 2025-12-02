import argparse
import torch
import os
from masked_diffusion import MaskedDiffusionTrainer
from model import UNet
from dataset import get_dataloader
from utils import save_images, load_checkpoint, save_checkpoint

def train(args):
    """Training function"""
    print(f"Initializing model...")
    model = UNet(
        in_channels=args.channels,
        out_channels=args.channels,
        image_size=args.image_size,
        base_channels=args.base_channels,
        time_embed_dim=args.time_embed_dim
    ).to(args.device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print(f"Initializing trainer...")
    trainer = MaskedDiffusionTrainer(
        model=model,
        image_size=args.image_size,
        channels=args.channels,
        device=args.device,
        num_timesteps=args.num_timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end
    )
    
    print(f"Loading dataset...")
    dataloader = get_dataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers
    )
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)
    
    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, trainer.optimizer)
        start_epoch = checkpoint['epoch']
        print(f"Resumed from epoch {start_epoch}")
    
    print(f"Starting training...")
    trainer.train(
        dataloader=dataloader,
        epochs=args.epochs,
        lr=args.lr,
        save_dir=args.save_dir,
        sample_dir=args.sample_dir,
        save_every=args.save_every,
        sample_every=args.sample_every,
        start_epoch=start_epoch
    )

def inference(args):
    """Inference function"""
    print(f"Loading model from {args.checkpoint}...")
    model = UNet(
        in_channels=args.channels,
        out_channels=args.channels,
        image_size=args.image_size,
        base_channels=args.base_channels,
        time_embed_dim=args.time_embed_dim
    ).to(args.device)
    
    trainer = MaskedDiffusionTrainer(
        model=model,
        image_size=args.image_size,
        channels=args.channels,
        device=args.device,
        num_timesteps=args.num_timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end
    )
    
    load_checkpoint(args.checkpoint, model)
    model.eval()
    
    print(f"Generating samples...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate samples with different mask types
    for mask_type in ['random', 'center', 'block']:
        print(f"Generating with {mask_type} mask...")
        samples = trainer.sample(
            num_samples=args.num_samples,
            mask_ratio=args.mask_ratio,
            mask_type=mask_type,
            guidance_scale=args.guidance_scale
        )
        save_images(samples, os.path.join(args.output_dir, f'samples_{mask_type}.png'))
    
    # Inpainting example if reference image provided
    if args.reference_image:
        print(f"Inpainting {args.reference_image}...")
        import torchvision.transforms as transforms
        from PIL import Image
        
        transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * args.channels, [0.5] * args.channels)
        ])
        
        img = Image.open(args.reference_image).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(args.device)
        
        # Create mask (example: mask center)
        mask = trainer.generate_mask(1, mask_ratio=args.mask_ratio, mask_type='center')
        
        inpainted = trainer.inpaint(img_tensor, mask, guidance_scale=args.guidance_scale)
        save_images(inpainted, os.path.join(args.output_dir, 'inpainted.png'))

def main():
    parser = argparse.ArgumentParser(description='Masked Diffusion Model')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'inference'],
                       help='Mode: train or inference')
    
    # Model parameters
    parser.add_argument('--image_size', type=int, default=32, help='Image size')
    parser.add_argument('--channels', type=int, default=3, help='Number of channels')
    parser.add_argument('--base_channels', type=int, default=64, help='Base channels in UNet')
    parser.add_argument('--time_embed_dim', type=int, default=128, help='Time embedding dimension')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_timesteps', type=int, default=1000, help='Number of diffusion timesteps')
    parser.add_argument('--beta_start', type=float, default=0.0001, help='Beta start')
    parser.add_argument('--beta_end', type=float, default=0.02, help='Beta end')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    
    # Save/load parameters
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--sample_dir', type=str, default='./samples', help='Directory to save samples')
    parser.add_argument('--save_every', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--sample_every', type=int, default=5, help='Generate samples every N epochs')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    # Inference parameters
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path for inference')
    parser.add_argument('--num_samples', type=int, default=16, help='Number of samples to generate')
    parser.add_argument('--mask_ratio', type=float, default=0.5, help='Mask ratio (ratio of pixels to preserve)')
    parser.add_argument('--mask_type', type=str, default='random', choices=['random', 'center', 'block'],
                       help='Mask type')
    parser.add_argument('--guidance_scale', type=float, default=1.0, help='Guidance scale')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory for inference')
    parser.add_argument('--reference_image', type=str, default=None, help='Reference image for inpainting')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(args)
    else:
        if args.checkpoint is None:
            raise ValueError("--checkpoint must be provided for inference mode")
        inference(args)

if __name__ == '__main__':
    main()

