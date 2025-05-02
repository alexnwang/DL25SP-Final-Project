# train.py

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np

import random
from typing import NamedTuple

from models import JEPA, init_momentum_params, update_momentum_params

# ----------------------------
# Configuration
# ----------------------------
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
repr_dim    = 256
proj_dim    = 256
lr          = 3e-4
epochs      = 5
batch_size  = 32
alpha       = 1.0
beta        = 0.01
temperature = 0.1
momentum    = 0.999

# ----------------------------
# Data loading (use memory map)
# ----------------------------
print("Loading data...")
states_np = np.load("/scratch/DL25SP/train/states.npy", mmap_mode="r")
actions_np = np.load("/scratch/DL25SP/train/actions.npy", mmap_mode="r")
states = torch.from_numpy(states_np).float()
actions = torch.from_numpy(actions_np).float()
print(f"Loaded {len(states)} trajectories.")

class Sample(NamedTuple):
    states: torch.Tensor
    actions: torch.Tensor

class TrajDataset(Dataset):
    def __init__(self, states, actions):
        self.states = states
        self.actions = actions

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return Sample(states=self.states[idx], actions=self.actions[idx])

train_loader = DataLoader(
    TrajDataset(states, actions),
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True,
)

# ----------------------------
# Augmentation: blur + flips
# ----------------------------
def gaussian_blur(frames: torch.Tensor) -> torch.Tensor:
    T, C, H, W = frames.shape
    kernel = torch.tensor([[1., 2., 1.],
                           [2., 4., 2.],
                           [1., 2., 1.]], device=frames.device) / 16.0
    kernel = kernel.view(1, 1, 3, 3).repeat(C, 1, 1, 1)
    x = frames.view(-1, C, H, W)
    x = F.conv2d(x, kernel, padding=1, groups=C)
    return x.view(T, C, H, W)

def simple_augment(batch: torch.Tensor) -> torch.Tensor:
    aug = batch.clone()
    B, T, C, H, W = aug.shape
    for i in range(B):
        if random.random() < 0.5:
            aug[i] = aug[i].flip(-1)  # horizontal
        if random.random() < 0.5:
            aug[i] = aug[i].flip(-2)  # vertical
        aug[i] = gaussian_blur(aug[i])
    return aug

# ----------------------------
# Model setup
# ----------------------------
model = JEPA(repr_dim=repr_dim, proj_dim=proj_dim).to(device)
model_m = JEPA(repr_dim=repr_dim, proj_dim=proj_dim).to(device)
init_momentum_params(model.encoder, model_m.encoder)
init_momentum_params(model.proj_head, model_m.proj_head)
for p in model_m.parameters():
    p.requires_grad = False

optimizer = optim.Adam(model.parameters(), lr=lr)

# ----------------------------
# Training loop
# ----------------------------
for epoch in range(1, epochs + 1):
    model.train()
    sum_jepa, sum_kl, sum_ctr = 0.0, 0.0, 0.0

    loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", unit="batch")
    for batch_idx, batch in enumerate(loop, 1):
        batch_states = batch.states.to(device)
        batch_actions = batch.actions.to(device)

        aug_states = simple_augment(batch_states)

        optimizer.zero_grad()
        jepa_l, kl_l, ctr_l = model.compute_loss(
            batch_states, batch_actions, aug_states, model_m,
            alpha=alpha, beta=beta, temperature=temperature
        )
        total_loss = jepa_l + beta * kl_l + alpha * ctr_l
        total_loss.backward()
        optimizer.step()

        update_momentum_params(model.encoder, model_m.encoder, momentum)
        update_momentum_params(model.proj_head, model_m.proj_head, momentum)

        sum_jepa += jepa_l.item()
        sum_kl += kl_l.item()
        sum_ctr += ctr_l.item()

        loop.set_postfix({
            "JEPA": f"{sum_jepa / batch_idx:.4f}",
            "KL": f"{sum_kl / batch_idx:.4f}",
            "CTR": f"{sum_ctr / batch_idx:.4f}",
        })

    print(f"Epoch {epoch} summary | JEPA: {sum_jepa/len(train_loader):.4f} | KL: {sum_kl/len(train_loader):.4f} | CTR: {sum_ctr/len(train_loader):.4f}")
    
    # Save model after each epoch
    torch.save(model.state_dict(), f"model_weights_epoch{epoch}.pth")
    print(f"Saved model_weights_epoch{epoch}.pth")


# ----------------------------
# Save model
# ----------------------------
torch.save(model.state_dict(), "model_weights.pth")
print("Saved model_weights.pth")
