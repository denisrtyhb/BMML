import torch

def get_alpha(t):
    """
    Linear Schedule.
    t is between 0 and 1.
    alpha = 1 - t.
    At t=0, alpha=1 (All visible).
    At t=1, alpha=0 (All masked).
    """
    return 1 - t

def generate_mask(x, t):
    """
    Generates a binary mask based on t.
    t: tensor [Batch_Size] of float values 0 to 1.
    
    Returns:
        mask: 1 if pixel is MASKED, 0 if VISIBLE.
    """
    b, c, h, w = x.shape
    device = x.device
    
    # Probability that a pixel is masked = t
    # Shape: [B, 1, 1, 1]
    mask_probs = t.view(-1, 1, 1, 1).expand(b, 1, h, w)
    
    random_matrix = torch.rand(b, 1, h, w, device=device)
    
    # If random < t, it gets masked (set to 1)
    # This ensures exactly proportion 't' is masked on average
    mask = (random_matrix < mask_probs).float()
    
    return mask

def get_loss_weight(t):
    """
    From the paper: weight ~ alpha' / (1 - alpha)
    For linear schedule: alpha' = -1, (1-alpha) = t.
    Weight = 1/t.
    """
    # Clamp to avoid division by zero
    return 1.0 / torch.clamp(t, min=1e-4)