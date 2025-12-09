# Masked Diffusion Language Model (MDLM) for MNIST

This repository contains code for training and sampling from masked diffusion models on binarized MNIST.

## Training

### Baseline Model

To train the baseline model, use `train_baseline.py`:

```bash
python train_baseline.py --n_epochs 10 --batch_size 64 --output_folder outputs
```

**Arguments:**
- `--n_epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch szie
- `--device`: Device to run on ('cuda' or 'cpu'). If not specified, auto-detects CUDA availability
- `--output_folder`: Folder to save model checkpoints (default: "outputs")

**Example:**
```bash
# Train for 20 epochs on GPU
python train_baseline.py --n_epochs 20 --batch_size 64 --device cuda --output_folder checkpoints

# Train for 10 epochs on CPU
python train_baseline.py --n_epochs 10 --batch_size 64 --device cpu
```

**Output:**
- Model checkpoints are saved every 5 epochs as `baseline_mnist_epoch{N}.pth`
- Final model is saved as `baseline_mnist_final.pth` in the specified output folder

### Ambient Model

To train the ambient model with consistency loss, use `train_ambient.py`:

```bash
python train_ambient.py --n_epochs 10 --output_folder outputs
```

**Arguments:**
- `--n_epochs`: Number of training epochs (default: 10)
- `--device`: Device to run on ('cuda' or 'cpu'). If not specified, auto-detects CUDA availability
- `--output_folder`: Folder to save model checkpoints (default: "outputs")
- `--consistency_weight`: Weight for consistency loss term (default: 1.0)

**Example:**
```bash
# Train with default consistency weight
python train_ambient.py --n_epochs 20 --device cuda --output_folder checkpoints

# Train with higher consistency weight
python train_ambient.py --n_epochs 20 --consistency_weight 2.0 --output_folder checkpoints
```

**Output:**
- Model checkpoints are saved every 5 epochs as `ambient_mnist_epoch{N}.pth`
- Final model is saved as `ambient_mnist_final.pth` in the specified output folder

**Note:** The ambient model uses a two-part loss:
- **Easy loss**: Standard denoising loss on higher noise levels
- **Hard loss**: Consistency loss that ensures the model makes consistent predictions at lower noise levels

## Sampling

To generate samples from a trained model, use `sample.py`:

```bash
python sample.py --model_path <path_to_checkpoint>
```

**Arguments:**
- `--model_path`: Path to model checkpoint (required)
- `--device`: Device to run on ('cuda' or 'cpu'). If not specified, auto-detects CUDA availability
- `--output_path`: Path to save generated samples (default: "ambient_generated.png")

**Examples:**
```bash
# Sample from baseline model
python sample.py --model_path outputs/baseline_mnist_final.pth

# Sample from ambient model
python sample.py --model_path outputs/ambient_mnist_final.pth --output_path ambient_samples.png

# Sample on CPU
python sample.py --model_path outputs/baseline_mnist_final.pth --device cpu
```

**Sampling Parameters (hardcoded in script):**
- `steps`: Number of denoising steps (default: 40, more steps = better quality)
- `b`: Batch size / number of samples to generate (default: 16)

**Output:**
- Generated samples are saved to the specified output path (default: `ambient_generated.png`)
- The script generates binary MNIST digits by iteratively unmasking pixels over multiple steps

## Requirements

The code requires:
- PyTorch
- torchvision
- tqdm
- numpy

## Dataset

The training uses `CorruptedMNIST` from `corrupt_data.py`, which provides MNIST images with a specified percentage of pixels masked (default: 10% observed, 90% masked).

