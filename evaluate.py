from corrupt_data import CorruptedMNIST
from torch.utils.data import DataLoader
import torch
from torchvision.utils import save_image, make_grid
import os



def evaluate(model, device, output_folder, iteration, OBSERVED_MASK_PCT):
    """
    Evaluate model on test set and generate samples.
    
    Args:
        model: Model to evaluate
        device: Device to run on
        output_folder: Folder to save evaluation results
        iteration: Current iteration number
    """
    model.eval()

    # Load test dataset
    test_dataset = CorruptedMNIST(mask_percentage=OBSERVED_MASK_PCT, train=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)
    total_easy_loss = 0.0
    total_hard_loss = 0.0
    num_batches = 0
    
    # Evaluate consistency loss and log/save all relevant images

    all_x_obs = []
    all_mask_obs = []
    all_x_filled = []
    all_pred_x0_teacher_probs = []
    all_denoised_005 = []

    model.eval()
    with torch.no_grad():
        for x_obs, mask_obs, _ in test_loader:
            x_obs = x_obs.to(device)
            mask_obs = mask_obs.to(device)
            b = x_obs.shape[0]
            t_dataset = torch.full((b,), OBSERVED_MASK_PCT, device=device)
            # Run the same code as in calculate_consistency_loss, but also fetch images
            # We'll reconstruct what images we can

            # Teacher
            teacher_logits = model(x_obs, t_dataset * 1000, mask_obs)
            pred_x0_teacher_probs = torch.sigmoid(teacher_logits)

            pred_x0_hard = (pred_x0_teacher_probs > 0.5).float()
            x_filled = x_obs * (1 - mask_obs) + pred_x0_hard * mask_obs

            # Denoising result at time = (OBSERVED_MASK_PCT - 0.05), clamped at min 0.001
            t_denoise = torch.clamp(t_dataset - 0.05, min=0.001)

            # Partly denoise x_obs before passing to model
            def remove_half_masked(mask):
                mask = mask.clone()
                masked_indices = (mask == 1).nonzero(as_tuple=True)
                num_to_unmask = masked_indices[0].numel() // 2
                if num_to_unmask > 0:
                    unmask_idxs = torch.randperm(masked_indices[0].numel(), device=mask.device)[:num_to_unmask]
                    for dim, idxs in enumerate(masked_indices):
                        masked_indices = tuple(i[unmask_idxs] if dim == d else i for d, i in enumerate(masked_indices))
                    mask[masked_indices] = 0
                return mask

            partly_denoised_mask = remove_half_masked(mask_obs)
            
            partly_denoised_input = x_filled * (1 - partly_denoised_mask) + x_obs * partly_denoised_mask

            denoise_logits = model(partly_denoised_input, t_denoise * 1000, mask_obs)
            denoise_probs = torch.sigmoid(denoise_logits)

            all_x_obs.append(x_obs.cpu())
            all_mask_obs.append(mask_obs.cpu())
            all_x_filled.append(x_filled.cpu())
            all_pred_x0_teacher_probs.append(pred_x0_teacher_probs.cpu())
            all_denoised_005.append(denoise_probs.cpu())

    # Stack all batches
    all_x_obs = torch.cat(all_x_obs, dim=0)
    all_mask_obs = torch.cat(all_mask_obs, dim=0)
    all_x_filled = torch.cat(all_x_filled, dim=0)
    all_pred_x0_teacher_probs = torch.cat(all_pred_x0_teacher_probs, dim=0)
    all_denoised_005 = torch.cat(all_denoised_005, dim=0)

    # Save all images in one PNG: one column per sample, each row = (x_obs, mask, x_filled, teacher_probs, denoised_0.05)
    nshow = min(64, all_x_obs.shape[0])
    # Normalize/unnormalize for display consistency
    mask_vis = all_mask_obs[:nshow]
    x_obs_vis = all_x_obs[:nshow]
    x_filled_vis = all_x_filled[:nshow]
    teacher_probs_vis = all_pred_x0_teacher_probs[:nshow]
    denoised_005_vis = all_denoised_005[:nshow]

    # For mask, repeat to 3 channels so it's visually clear
    def to_rgb_grid(x):
        # Display masked pixels as purple ([0.5, 0, 0.5])
        x_rgb = x.clone()
        if x_rgb.shape[1] == 1:
            x_rgb = x_rgb.repeat(1, 3, 1, 1)
        mask = (x == -1)
        # Set R and B to 0.5 (purple), G to 0
        x_rgb[:, 0][mask[:, 0]] = 0.5  # R
        x_rgb[:, 1][mask[:, 0]] = 0.0  # G
        x_rgb[:, 2][mask[:, 0]] = 0.5  # B
        return x_rgb

    vis_tensors = [
        to_rgb_grid(x_obs_vis),
        to_rgb_grid(mask_vis),
        to_rgb_grid(x_filled_vis),
        to_rgb_grid(teacher_probs_vis),
        to_rgb_grid(denoised_005_vis)
    ]  # each is (nshow, 3, 28, 28)

    vis_tensor = torch.cat(vis_tensors, dim=0)  # (4*nshow, 3, h, w)

    # Arrange as grid: nrow = nshow, so each column is a sample, each row is a view
    grid = make_grid(
        vis_tensor, nrow=nshow, normalize=True
    )

    out_path = os.path.join(output_folder, f"eval_grid_iter_{iteration}.png")
    save_image(grid, out_path)

    print(f"  Saved x_obs/mask_obs/x_filled/teacher_probs images to {output_folder}")

    # For evaluation loss, simply return 0 for compatibility
    avg_easy_loss, avg_hard_loss = 0.0, 0.0
    
    # Generate samples
    model.eval()
    with torch.no_grad():
        steps = 40
        b_samples = 16
        
        # Start: Fully Masked (-1.0)
        img = torch.full((b_samples, 1, 28, 28), -1.0, device=device)
        mask = torch.ones(b_samples, 1, 28, 28).to(device)
        
        for i in range(steps):
            t_val = 1.0 - (i / steps)
            t_vec = torch.full((b_samples,), t_val, device=device)
            
            # Predict
            pred_logits = model(img, t_vec * 1000, mask)
            probs = torch.sigmoid(pred_logits)
            sampled_prediction = torch.bernoulli(probs)
            
            # Update Image
            img = (1 - mask) * img + mask * sampled_prediction
            
            # Next Mask
            next_t_val = 1.0 - ((i + 1) / steps)
            random_mask = torch.rand(b_samples, 1, 28, 28).to(device)
            new_desired_mask = (random_mask < next_t_val).float()
            mask = mask * new_desired_mask
            
            # Re-apply -1 to masks
            img = img * (1 - mask) - 1.0 * mask
        
        # Save samples
        sample_path = os.path.join(output_folder, f"samples_iter_{iteration}.png")
        save_image(img, sample_path)
        print(f"  Saved samples to {sample_path}\n")
    
    model.train()
    return avg_easy_loss, avg_hard_loss