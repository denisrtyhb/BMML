import torch

def get_alpha(t):
    """
    Linear schedule: Signal alpha starts at 1 (clean), goes to 0 (noise).
    alpha(t) = 1 - t
    """
    return 1 - t

def generate_mask(x, t):
    """
    Generates a binary mask where 't' percent of pixels are 1 (Masked).
    """
    b, c, h, w = x.shape
    device = x.device
    
    # Probability that a pixel is masked = t
    mask_probs = t.view(-1, 1, 1, 1).expand(b, 1, h, w)
    
    random_matrix = torch.rand(b, 1, h, w, device=device)
    
    # 1 = Masked, 0 = Visible
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