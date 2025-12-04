# import torch

# def get_alpha(t):
#     """
#     Linear Schedule.
#     t is between 0 and 1.
#     alpha = 1 - t.
#     At t=0, alpha=1 (All visible).
#     At t=1, alpha=0 (All masked).
#     """
#     return 1 - t

# def generate_mask(x, t):
#     """
#     Generates a binary mask based on t.
#     t: tensor [Batch_Size] of float values 0 to 1.
    
#     Returns:
#         mask: 1 if pixel is MASKED, 0 if VISIBLE.
#     """
#     b, c, h, w = x.shape
#     device = x.device
    
#     # Probability that a pixel is masked = t
#     # Shape: [B, 1, 1, 1]
#     mask_probs = t.view(-1, 1, 1, 1).expand(b, 1, h, w)
    
#     random_matrix = torch.rand(b, 1, h, w, device=device)
    
#     # If random < t, it gets masked (set to 1)
#     # This ensures exactly proportion 't' is masked on average
#     mask = (random_matrix < mask_probs).float()
    
#     return mask

# def get_loss_weight(t):
#     """
#     From the paper: weight ~ alpha' / (1 - alpha)
#     For linear schedule: alpha' = -1, (1-alpha) = t.
#     Weight = 1/t.
#     """
#     # Clamp to avoid division by zero
#     return 1.0 / torch.clamp(t, min=1e-4)
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

def get_nelbo_weight(t):
    """
    Calculates the weight term for the NELBO loss.
    
    Formula: Weight = |alpha'(t)| / (1 - alpha(t))
    
    For Linear Schedule (alpha = 1 - t):
       alpha' = -1
       1 - alpha = t
       Weight = |-1| / t = 1/t
    """
    # Clamp t to a tiny minimum (1e-5) to avoid dividing by zero
    # This prevents the loss from exploding to Infinity at t=0
    t_clamped = torch.clamp(t, min=1e-5)
    
    weights = 1.0 / t_clamped
    return weights