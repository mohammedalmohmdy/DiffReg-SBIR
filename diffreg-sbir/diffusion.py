import torch
import math

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps
    x = torch.linspace(0, steps, steps+1)
    alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clamp(betas, 0.0001, 0.999)
    return betas

def q_sample(z0, t, betas, device):
    # z0: [B, D], t: integer tensor [B] in [0,T-1]
    sqrt_alphas_cumprod = torch.sqrt(1.0 - betas.cumsum(0))  # simplified
    noise = torch.randn_like(z0).to(device)
    # simple per-sample scaling (approximate)
    alpha = (1.0 - betas.mean()).to(device)
    return z0 * alpha + noise * (1 - alpha), noise