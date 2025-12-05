# Masked Diffusion Language Model (MDLM) for MNIST

This repository contains code for training and sampling from masked diffusion models on binarized MNIST.

## Training

### Baseline Model

To train the baseline model, use `train_baseline.py`:

```bash
python train_baseline.py --n_epochs 10 --output_folder outputs
```

**Arguments:**
- `--n_epochs`: Number of training epochs (default: 10)
- `--device`: Device to run on ('cuda' or 'cpu'). If not specified, auto-detects CUDA availability
- `--output_folder`: Folder to save model checkpoints (default: "outputs")

**Example:**
```bash
# Train for 20 epochs on GPU
python train_baseline.py --n_epochs 20 --device cuda --output_folder checkpoints

# Train for 10 epochs on CPU
python train_baseline.py --n_epochs 10 --device cpu
```

**Output:**
- Model checkpoints are saved every 5 epochs as `baseline_mnist_epoch{N}.pth`
- Final model is saved as `baseline_mnist_final.pth` in the specified output folder

## Sampling

To generate samples from a trained model, use `sample.py`:

```bash
python sample.py
```

**Note:** Before running `sample.py`, you need to update the model path in the script to point to your trained checkpoint. 

The script currently tries to load:
- `mdlm_ambient_consistency.pth` (or `/kaggle/working/simple_gpt_try/mdlm_ambient_consistency.pth`)

To use a baseline model checkpoint, modify `sample.py` around line 216 to load your checkpoint:

```python
# In sample.py, change the model loading line:
model.load_state_dict(torch.load("outputs/baseline_mnist_final.pth"))
```

**Sampling Parameters:**
- `steps`: Number of denoising steps (default: 40, more steps = better quality)
- `b`: Batch size / number of samples to generate (default: 16)

**Output:**
- Generated samples are saved as `ambient_generated.png` (or `mdlm_binary_result.png` depending on the version)
- The script generates binary MNIST digits by iteratively unmasking pixels over multiple steps

## Requirements

The code requires:
- PyTorch
- torchvision
- tqdm
- numpy

## Dataset

The training uses `CorruptedMNIST` from `corrupt_data.py`, which provides MNIST images with a specified percentage of pixels masked (default: 10% observed, 90% masked).

