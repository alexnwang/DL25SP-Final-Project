from typing import List
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch


class Encoder(nn.Module):
    def __init__(self, repr_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, 4, stride=2, padding=1),  # 64x64 -> 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 16x16 -> 8x8
            nn.ReLU(),
            nn.Flatten(),  # 128*8*8 = 8192
            nn.Linear(8192, repr_dim),
        )

    def forward(self, x):
        return self.conv(x)

class Predictor(nn.Module):
    def __init__(self, repr_dim=256, action_dim=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(repr_dim + action_dim, 512),
            nn.ReLU(),
            nn.Linear(512, repr_dim),
        )

    def forward(self, h, a):
        # h: (batch, repr_dim), a: (batch, action_dim)
        x = torch.cat([h, a], dim=-1)
        return self.fc(x)

class JEPA(nn.Module):
    def __init__(self, repr_dim=256):
        super().__init__()
        self.encoder = Encoder(repr_dim=repr_dim)
        self.predictor = Predictor(repr_dim=repr_dim)
        self.repr_dim = repr_dim


    def forward(self, states, actions):
        """
        Args:
            states: (batch, 1, C, H, W)  # Only the first observation
            actions: (batch, T, 2)        # Sequence of actions
    
        Output:
            predictions: (batch, T, repr_dim)
        """
        B, T, _ = actions.shape
    
        # Encode the initial observation
        init_obs = states[:, 0]  # (B, C, H, W)
        s = self.encoder(init_obs)  # (B, repr_dim)
    
        preds = []
    
        for t in range(T):
            a = actions[:, t]  # (B, 2)
            s = self.predictor(s, a)  # (B, repr_dim)
            preds.append(s)
    
        preds = torch.stack(preds, dim=1)  # (B, T, repr_dim)
    
        return preds


    def compute_loss(self, states, actions):
        """
        Energy loss: || s~_t - Enc(o_t) ||^2
        """
        with torch.no_grad():
            target_states = torch.stack([self.encoder(states[:, t]) for t in range(states.shape[1])], dim=1)

        preds = self.forward(states, actions)
        loss = F.mse_loss(preds, target_states[:, 1:])
        return loss


def build_mlp(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)


class MockModel(torch.nn.Module):
    """
    Does nothing. Just for testing.
    """

    def __init__(self, device="cuda", output_dim=256):
        super().__init__()
        self.device = device
        self.repr_dim = output_dim

    def forward(self, states, actions):
        """
        Args:
            During training:
                states: [B, T, Ch, H, W]
            During inference:
                states: [B, 1, Ch, H, W]
            actions: [B, T-1, 2]

        Output:
            predictions: [B, T, D]
        """
        B, T, _ = actions.shape

        return torch.randn((B, T + 1, self.repr_dim)).to(self.device)


class Prober(torch.nn.Module):
    def __init__(
        self,
        embedding: int,
        arch: str,
        output_shape: List[int],
    ):
        super().__init__()
        self.output_dim = np.prod(output_shape)
        self.output_shape = output_shape
        self.arch = arch

        arch_list = list(map(int, arch.split("-"))) if arch != "" else []
        f = [embedding] + arch_list + [self.output_dim]
        layers = []
        for i in range(len(f) - 2):
            layers.append(torch.nn.Linear(f[i], f[i + 1]))
            layers.append(torch.nn.ReLU(True))
        layers.append(torch.nn.Linear(f[-2], f[-1]))
        self.prober = torch.nn.Sequential(*layers)

    def forward(self, e):
        output = self.prober(e)
        return output

