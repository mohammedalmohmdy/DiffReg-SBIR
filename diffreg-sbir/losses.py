import torch
import torch.nn.functional as F

def triplet_margin_loss(anchor, positive, negative, margin=0.2):
    pos = F.pairwise_distance(anchor, positive)
    neg = F.pairwise_distance(anchor, negative)
    loss = F.relu(pos - neg + margin).mean()
    return loss

def l2_loss(a,b):
    return F.mse_loss(a,b)