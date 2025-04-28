# train.py

import torch
import torch.optim as optim
import random
import numpy as np
from typing import NamedTuple
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn

from models import (
    JEPA,
    nt_xent_loss,
    init_momentum_params,
    update_momentum_params,
)

# Hyperparameters
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size  = 16
epochs      = 10
lr          = 1e-3
alpha       = 1.0
beta        = 0.1
temperature = 0.5
momentum    = 0.999

# -------------------------------------------------------------------
# Data loading: memory-map the big .npy files, take a slice, wrap in a Dataset
# -------------------------------------------------------------------
print("Loading data (first 1000 trajectories) with mmap...")
states_np  = np.load('/scratch/DL25SP/train/states.npy',  mmap_mode='r')[:5000]  # (N, T, 2, H, W)
actions_np = np.load('/scratch/DL25SP/train/actions.npy', mmap_mode='r')[:5000]  # (N, T-1, 2)
print("Data loaded.")

# Convert to torch (on CPU)
states_all  = torch.from_numpy(states_np).float()
actions_all = torch.from_numpy(actions_np).float()

# NamedTuple + Dataset so that DataLoader yields batch.states & batch.actions
class Sample(NamedTuple):
    states: torch.Tensor
    actions: torch.Tensor

class MemDataset(Dataset):
    def __init__(self, states, actions):
        self.states  = states
        self.actions = actions
    def __len__(self):
        return len(self.states)
    def __getitem__(self, idx):
        return Sample(states=self.states[idx], actions=self.actions[idx])

train_loader = DataLoader(
    MemDataset(states_all, actions_all),
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True,
)

# -------------------------------------------------------------------
# Models + momentum copy
# -------------------------------------------------------------------
model   = JEPA(repr_dim=256, proj_dim=256).to(device)
model_m = JEPA(repr_dim=256, proj_dim=256).to(device)
init_momentum_params(model.encoder,   model_m.encoder)
init_momentum_params(model.proj_head, model_m.proj_head)
for p in model_m.parameters():
    p.requires_grad = False

optimizer = optim.Adam(model.parameters(), lr=lr)

# -------------------------------------------------------------------
# 3) Prepare Gaussian kernel for blur
# -------------------------------------------------------------------
# 3x3 Gaussian kernel weights (unnormalized): [1 2 1; 2 4 2; 1 2 1] / 16
gauss_kernel = torch.tensor(
    [[1.0, 2.0, 1.0],
     [2.0, 4.0, 2.0],
     [1.0, 2.0, 1.0]],
    device=device
) / 16.0
# we have 2 image channels, so create a weight tensor of shape [C,1,3,3]
gauss_weight = gauss_kernel.view(1,1,3,3).repeat(2,1,1,1)

def gaussian_blur(frames: torch.Tensor) -> torch.Tensor:
    """
    frames: [T, C, H, W]
    returns: blurred frames, same shape
    """
    T, C, H, W = frames.shape
    x = frames.view(-1, C, H, W)  # [T, C, H, W]
    # apply depthwise conv2d with our Gaussian kernel
    x_blur = F.conv2d(x, weight=gauss_weight, bias=None,
                      stride=1, padding=1, groups=C)
    return x_blur.view(T, C, H, W)

# -------------------------------------------------------------------
# 4) Augmentation: crop, jitter, blur, flips, rotations
# -------------------------------------------------------------------
def augment_batch(batch_states: torch.Tensor) -> torch.Tensor:
    """
    batch_states: [B, T, C, H, W] on `device`
    Returns an augmented copy of shape [B, T, C, H, W].
    """
    aug = batch_states.clone()
    B, T, C, H, W = aug.shape

    for i in range(B):
        # a) Random resized crop (80–100%), then resize back
        scale = random.uniform(0.8, 1.0)
        new_h, new_w = int(H*scale), int(W*scale)
        top  = random.randint(0, H - new_h)
        left = random.randint(0, W - new_w)

        # Crop & resize each frame
        crop = aug[i, :, :, top:top+new_h, left:left+new_w]   # [T,C,new_h,new_w]
        crop = crop.view(-1, C, new_h, new_w)                  # [T, C, new_h, new_w]
        resized = F.interpolate(crop, size=(H, W),
                                mode='bilinear',
                                align_corners=False)          # [T, C, H, W]
        aug[i] = resized.view(T, C, H, W)

        # b) Brightness & contrast jitter
        brightness = random.uniform(0.8, 1.2)
        contrast   = random.uniform(0.8, 1.2)
        mean = aug[i].mean(dim=[2,3], keepdim=True)            # [T,C,1,1]
        aug[i] = (aug[i] - mean) * contrast + mean
        aug[i] = aug[i] * brightness

        # c) Gaussian blur
        aug[i] = gaussian_blur(aug[i])                         # [T,C,H,W]

        # d) Random flips & 90° rotations
        if random.random() < 0.5:
            aug[i] = aug[i].flip(-1)   # horiz
        if random.random() < 0.5:
            aug[i] = aug[i].flip(-2)   # vert
        k = random.randint(0, 3)
        aug[i] = torch.rot90(aug[i], k, dims=(-2, -1))

    return aug

# -------------------------------------------------------------------
# Training loop with tqdm (fixed division logic)
# -------------------------------------------------------------------
for epoch in range(1, epochs + 1):
    model.train()
    sum_j, sum_kl, sum_ctr = 0.0, 0.0, 0.0
    n_batches = len(train_loader)

    loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", unit="batch")
    for batch_idx, batch in enumerate(loop, start=1):
        states  = batch.states.to(device)   # [B, T, 2, H, W]
        actions = batch.actions.to(device)  # [B, T-1, 2]

        aug_states = augment_batch(states)

        optimizer.zero_grad()
        jepa_l, kl_l, ctr_l = model.compute_loss(
            states, actions, aug_states, model_m,
            alpha=alpha, beta=beta, temperature=temperature
        )
        loss = jepa_l + beta * kl_l + alpha * ctr_l
        loss.backward()
        optimizer.step()

        update_momentum_params(model.encoder,   model_m.encoder,   momentum)
        update_momentum_params(model.proj_head, model_m.proj_head, momentum)

        sum_j  += jepa_l.item()
        sum_kl += kl_l.item()
        sum_ctr+= ctr_l.item()

        # average up to this batch
        loop.set_postfix({
            "JEPA": f"{sum_j/batch_idx:.4f}",
            "KL":   f"{sum_kl/batch_idx:.4f}",
            "CTR":  f"{sum_ctr/batch_idx:.4f}"
        })

    # epoch summary
    print(
        f"Epoch {epoch}/{epochs} | "
        f"JEPA {sum_j/n_batches:.4f} | "
        f"KL {sum_kl/n_batches:.4f} | "
        f"CTR {sum_ctr/n_batches:.4f}"
    )

# Save only the online model
torch.save(model.state_dict(), "model_weights.pth")
print("Saved model_weights.pth")

