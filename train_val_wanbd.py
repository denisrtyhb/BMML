import wandb
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from unet import UNet
from corrupt_data import CorruptedMNIST
import argparse
import os
import matplotlib.pyplot as plt

# Argument parser setup
parser = argparse.ArgumentParser(description="Train baseline model on CorruptedMNIST")
parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs to train for")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--device", type=str, default=None, help="Device to run on ('cuda', 'cpu'); if None, auto-detect")
parser.add_argument("--api_wanbd", type=str, default=None, help="API key for logging the training")
parser.add_argument("--output_folder", type=str, default="outputs", help="Folder to save outputs/checkpoints")

args = parser.parse_args()

n_epochs = getattr(args, "n_epochs", 10)
api_key = getattr(args, "api_wanbd", 'key')
batch_size = getattr(args, "batch_size", 64)
output_folder = getattr(args, "output_folder", "outputs")
device = getattr(args, "device", 'cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

os.makedirs(output_folder, exist_ok=True)

def save_sample_images(original_images, predicted_images, epoch):
    # Create a plot for the sample images
    fig, axs = plt.subplots(2, len(original_images), figsize=(12, 6))
    
    for i in range(len(original_images)):
        # Plot original images (top row)
        axs[0, i].imshow(original_images[i].cpu().numpy().squeeze(), cmap="gray")
        axs[0, i].axis('off')
        axs[0, i].set_title(f"Original {i+1}")
        
        # Plot predicted images (bottom row)
        axs[1, i].imshow(predicted_images[i].cpu().detach().numpy().squeeze(), cmap="gray")
        axs[1, i].axis('off')
        axs[1, i].set_title(f"Pred {i+1}")

    # Save the figure and log it to wandb
    image_path = os.path.join(output_folder, f"epoch_{epoch}_sample_images.png")
    plt.tight_layout()
    plt.savefig(image_path)
    wandb.log({"validation_samples": wandb.Image(image_path)})
    plt.close(fig)

def train():
    # Initialize W&B logging
    os.environ['WANDB_API_KEY'] = api_key
    wandb.init(
        project="corrupted-mnist-baseline",
        config={
            "epochs": n_epochs,
            "batch_size": batch_size,
            "obs_mask_pct": 0.1,
            "lr": 1e-4,
            "device": device,
        }
    )

    OBSERVED_MASK_PCT = wandb.config.obs_mask_pct
    dataset = CorruptedMNIST(mask_percentage=OBSERVED_MASK_PCT, train=True)
    loader = DataLoader(dataset, batch_size=wandb.config.batch_size, shuffle=True, num_workers=2)
    val_dataset = CorruptedMNIST(mask_percentage=OBSERVED_MASK_PCT, train=False)
    val_loader = DataLoader(val_dataset, batch_size=5, shuffle=False)

    model = UNet().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=wandb.config.lr)

    # Track gradients and parameters of the model with W&B
    wandb.watch(model, log="all", log_freq=100)

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{n_epochs}")
        for x_obs, mask_obs, _ in pbar:
            x_obs = x_obs.to(device)
            mask_obs = mask_obs.to(device)
            b = x_obs.shape[0]

            t_easy = torch.rand(b, device=device) * (1.0 - OBSERVED_MASK_PCT) + OBSERVED_MASK_PCT
            mask_probs = t_easy.view(-1, 1, 1, 1).expand(b, 1, 28, 28)
            extra_mask = (torch.rand_like(x_obs) < mask_probs).float()

            final_mask = torch.max(mask_obs, extra_mask)

            x_input = x_obs.clone()
            x_input[extra_mask == 1] = -1.0

            pred_logits = model(x_input, t_easy * 1000, final_mask)

            learnable_region = (final_mask == 1) & (mask_obs == 0)

            loss = F.binary_cross_entropy_with_logits(pred_logits, x_obs, reduction='none')
            loss = (loss * learnable_region).sum() / (learnable_region.sum() + 1e-6)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            loss_val = loss.item()
            epoch_loss += loss_val

            # Log batch loss and model parameters
            wandb.log({"batch_loss": loss_val})

            pbar.set_postfix(loss=loss_val)

        avg_epoch_loss = epoch_loss / len(loader)
        wandb.log({"epoch_loss": avg_epoch_loss}, step=epoch+1)

        # Validation every 2 epochs
        if (epoch + 1) % 2 == 0:
            model.eval()
            with torch.no_grad():
                val_batch = next(iter(val_loader))
                x_val, mask_val, _ = val_batch
                x_val = x_val.to(device)
                mask_val = mask_val.to(device)

                # Mask input and predict
                t_easy = torch.rand(x_val.shape[0], device=device) * (1.0 - OBSERVED_MASK_PCT) + OBSERVED_MASK_PCT
                mask_probs = t_easy.view(-1, 1, 1, 1).expand(x_val.shape[0], 1, 28, 28)
                extra_mask = (torch.rand_like(x_val) < mask_probs).float()

                final_mask = torch.max(mask_val, extra_mask)
                x_input = x_val.clone()
                x_input[extra_mask == 1] = -1.0

                pred_logits = model(x_input, t_easy * 1000, final_mask)

                # Log validation images and predictions
                save_sample_images(x_val, pred_logits, epoch + 1)

        # Optionally log images during training (e.g., model predictions)
        if (epoch + 1) % 5 == 0:
            wandb.log({
                "sample_images": [wandb.Image(x_obs[0].cpu(), caption="Input Image"), 
                                  wandb.Image(pred_logits[0].cpu(), caption="Predicted Image")]
            })

        # Save the model checkpoints
        if (epoch + 1) % 5 == 0:
            model_path = os.path.join(output_folder, f"baseline_mnist_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), model_path)
            wandb.save(model_path)

    final_path = os.path.join(output_folder, "baseline_mnist_final.pth")
    torch.save(model.state_dict(), final_path)
    wandb.save(final_path)

    print("Training Done.")
    wandb.finish()

if __name__ == "__main__":
    train()
