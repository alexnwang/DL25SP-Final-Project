import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from models import JEPA, nt_xent_loss
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

print("Loading data...")
states_np = np.load('/scratch/DL25SP/train/states.npy', mmap_mode = 'r')[:1000]  # (N, T, 2, H, W)
actions_np = np.load('/scratch/DL25SP/train/actions.npy', mmap_mode = 'r')[:1000]  # (N, T-1, 2)
states = torch.tensor(states_np, dtype=torch.float32)
actions = torch.tensor(actions_np, dtype=torch.float32)
print("Data loaded.")

def augment_batch(batch_states):
    """
    Apply random flips and rotations to each trajectory in the batch.
    batch_states: Tensor [B, T, C, H, W]
    """
    aug = batch_states.clone()
    B, T, C, H, W = aug.shape
    for i in range(B):
        # random horizontal flip
        if random.random() < 0.5:
            aug[i] = aug[i].flip(-1)
        # random vertical flip
        if random.random() < 0.5:
            aug[i] = aug[i].flip(-2)
        # random 0,90,180,270 rotation
        k = random.randint(0, 3)
        aug[i] = torch.rot90(aug[i], k, dims=(-2, -1))
    return aug

# Hyperparameters
epochs = 50
batch_size = 32
alpha = 1.0          # contrastive loss weight
temperature = 0.5    # for nt_xent_loss

model = JEPA().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

num_samples = states.shape[0]
indices = list(range(num_samples))

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    random.shuffle(indices)

    for start in range(0, num_samples, batch_size):
        idx = indices[start:start+batch_size]
        batch_states = states[idx].to(device)    # [B, T, 2, H, W]
        batch_actions = actions[idx].to(device)  # [B, T-1, 2]

        optimizer.zero_grad()

        # JEPA loss
        jepa_loss = model.compute_loss(batch_states, batch_actions)

        # Contrastive loss
        aug_states = augment_batch(batch_states)
        B, T, C, H, W = batch_states.shape
        orig_flat = batch_states.reshape(-1, C, H, W)
        aug_flat = aug_states.reshape(-1, C, H, W)

        with torch.no_grad():
            # get embeddings without tracking grad for original for speed
            orig_emb = model.encoder(orig_flat)
        aug_emb = model.encoder(aug_flat)  # [B*T, D]

        contrastive_loss = nt_xent_loss(orig_emb, aug_emb, temperature)

        loss = jepa_loss + alpha * contrastive_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / (num_samples // batch_size)
    print(f"Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.6f}")

# Save
torch.save(model.state_dict(), "model_weights.pth")
print("Saved model_weights.pth")

