# Masked Diffusion Model

Implementation of Masked Diffusion for image generation and inpainting.

## Features

- **Masked Diffusion Training**: Train diffusion models with masking to focus learning on specific regions
- **Inpainting**: Fill in masked regions of images
- **Flexible Masking**: Support for random, center, and block masks
- **Colored MNIST Dataset**: Includes colored MNIST dataset for easy experimentation

## Installation

```bash
pip install torch torchvision pillow matplotlib numpy tqdm
```

## Usage

### Training

Train a masked diffusion model:

```bash
python main.py --mode train \
    --batch_size 64 \
    --epochs 100 \
    --lr 1e-4 \
    --image_size 32 \
    --save_dir ./checkpoints \
    --sample_dir ./samples
```

### Inference

Generate samples:

```bash
python main.py --mode inference \
    --checkpoint ./checkpoints/checkpoint_epoch_100.pth \
    --num_samples 16 \
    --mask_ratio 0.5 \
    --mask_type random \
    --output_dir ./outputs
```

### Inpainting

Inpaint a specific image:

```bash
python main.py --mode inference \
    --checkpoint ./checkpoints/checkpoint_epoch_100.pth \
    --reference_image path/to/image.png \
    --mask_ratio 0.3 \
    --output_dir ./outputs
```

## Arguments

### Training Arguments
- `--mode`: `train` or `inference`
- `--batch_size`: Batch size (default: 64)
- `--epochs`: Number of training epochs (default: 100)
- `--lr`: Learning rate (default: 1e-4)
- `--image_size`: Image size (default: 32)
- `--channels`: Number of channels (default: 3)
- `--num_timesteps`: Number of diffusion timesteps (default: 1000)
- `--data_dir`: Data directory (default: ./data)
- `--save_dir`: Checkpoint save directory (default: ./checkpoints)
- `--sample_dir`: Sample save directory (default: ./samples)

### Inference Arguments
- `--checkpoint`: Path to model checkpoint
- `--num_samples`: Number of samples to generate
- `--mask_ratio`: Ratio of pixels to preserve (default: 0.5)
- `--mask_type`: Type of mask: `random`, `center`, or `block`
- `--output_dir`: Output directory for generated images
- `--reference_image`: Path to image for inpainting

## Architecture

The model uses a UNet architecture that takes:
- Noisy image
- Timestep embedding
- Mask (binary mask indicating which regions to preserve/generate)

## Mask Types

- **random**: Random pixel masking
- **center**: Center block masking
- **block**: Random block masking

## Files

- `main.py`: Entry point script
- `masked_diffusion.py`: Core masked diffusion implementation
- `model.py`: UNet model architecture
- `dataset.py`: Data loading utilities
- `utils.py`: Helper functions

