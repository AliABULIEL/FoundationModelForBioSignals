import random
import torch
import torch.nn.functional as F
import numpy as np


class BiosignalAugment:
    def __init__(self, crop_p=0.5, noise_p=0.5, time_warp_p=0.3,
                 mag_warp_p=0.3, cutout_p=0.4, cutout_frac=0.1):
        self.crop_p = crop_p
        self.noise_p = noise_p
        self.time_warp_p = time_warp_p
        self.mag_warp_p = mag_warp_p
        self.cutout_p = cutout_p
        self.cutout_frac = cutout_frac

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure tensor is float32 for MPS compatibility
        x = x.float()
        C, L = x.shape

        if random.random() < self.crop_p:
            shift = random.randint(0, int(0.1 * L))
            x = torch.roll(x, -shift, dims=1)

        if random.random() < self.noise_p:
            x = x + 0.005 * torch.randn_like(x)

        if random.random() < self.time_warp_p:
            factor = 1 + 0.2 * (random.random() - 0.5)
            # Use F.interpolate for time warping
            if factor != 1.0:
                # Ensure x is float32 before interpolation
                x = x.float()
                # Add batch dimension for interpolate
                x_expanded = x.unsqueeze(0)  # [1, C, L]
                new_length = max(1, int(L * factor))
                x_warped = F.interpolate(x_expanded, size=new_length, mode='linear', align_corners=False)
                x = x_warped.squeeze(0)  # [C, new_L]

                # Crop or pad back to original length
                if x.shape[1] > L:
                    start = random.randint(0, x.shape[1] - L)
                    x = x[:, start:start + L]
                elif x.shape[1] < L:
                    pad = L - x.shape[1]
                    x = F.pad(x, (0, pad), value=0.0)

        if random.random() < self.mag_warp_p:
            warp = 1 + 0.1 * torch.randn(1, L, device=x.device)
            x = x * warp

        if random.random() < self.cutout_p:
            cut_len = int(self.cutout_frac * L)
            start = random.randint(0, L - cut_len)
            x[:, start:start + cut_len] = 0.0

        return x