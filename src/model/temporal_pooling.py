import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalPooling(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        # x: [B, T, d]
        # Average pooling over time
        x_avg = x.mean(dim=1) # [B, d]
        
        # L2 Normalize
        x_norm = F.normalize(x_avg, p=2, dim=1)
        
        return x_norm
