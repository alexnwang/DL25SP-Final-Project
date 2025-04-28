# train.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models import JEPA

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

print("Starting to load data...") 

# Load the training dataset
states = np.load('/scratch/DL25SP/train/states.npy')[:1000]  # shape: (num_trajectories, trajectory_len, 2, 64, 64)
actions = np.load('/scratch/DL25SP/train/actions.npy')[:1000]  # shape: (num_trajectories, trajectory_len-1, 2)

# Convert to torch tensors
states = torch.tensor(states, dtype=torch.float32).to(device)
actions = torch.tensor(actions, dtype=torch.float32).to(device)

print("Finished loading data. Starting training now...")  # <-- AFTER loading

# Initialize JEPA model
model = JEPA().to(device)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
epochs = 5  # Start with something small
batch_size = 32
num_trajectories = states.shape[0]

for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    # Shuffle indices
    idxs = torch.randperm(num_trajectories)

    for i in range(0, num_trajectories, batch_size):
        batch_idxs = idxs[i:i+batch_size]
        batch_states = states[batch_idxs]  # (batch, T, 2, 64, 64)
        batch_actions = actions[batch_idxs]  # (batch, T-1, 2)

        optimizer.zero_grad()
        loss = model.compute_loss(batch_states, batch_actions)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / (num_trajectories // batch_size)
    print(f"Epoch {epoch+1} / {epochs} | Avg Loss: {avg_loss:.6f}")

# Save model weights
torch.save(model.state_dict(), "model_weights.pth")
print("Model saved as model_weights.pth")
