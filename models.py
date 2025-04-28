from typing import List
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.inception import BasicConv2d
import numpy as np


def build_mlp(layers_dims: list[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)


class InceptionEncoder(nn.Module):
    """
    Encodes 2-channel observations into a repr_dim-dimensional vector
    using Inception v3 architecture with random initialization.
    """
    def __init__(self, repr_dim: int = 256, in_channels: int = 2):
        super().__init__()
        # Load InceptionV3 with random weights (no pretrained)
        inception = models.inception_v3(
            weights=None,
            aux_logits=False,
            transform_input=False
        )
        # Adapt first conv to accept in_channels
        inception.Conv2d_1a_3x3 = BasicConv2d(in_channels, 32, kernel_size=3, stride=2)

        # Feature extraction layers
        self.features = nn.Sequential(
            inception.Conv2d_1a_3x3,    # -> [B,32,H/2,W/2]
            inception.Conv2d_2a_3x3,    # -> [B,32,H/2-2,W/2-2]
            inception.Conv2d_2b_3x3,    # -> [B,64,H/2-2,W/2-2]
            nn.MaxPool2d(3, stride=2),  # -> [B,64,((H/2-2)-1)//2+1,((W/2-2)-1)//2+1]
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(3, stride=2),
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()                # -> [B,2048]
        )
        self.projection = nn.Linear(2048, repr_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,2,65,65] upsample to 299x299
        x = F.interpolate(x, size=(128, 128), mode='bilinear', align_corners=False)
        # now safe to run full Inception pipeline
        x = self.features(x)           # [B,2048,1,1]->[B,2048]
        return self.projection(x)      # [B,repr_dim]

class Predictor(nn.Module):
    def __init__(self, repr_dim: int = 256, action_dim: int = 2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(repr_dim + action_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, repr_dim)
        )

    def forward(self, h: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        # h: [B,repr_dim], a: [B,action_dim]
        return self.fc(torch.cat([h, a], dim=-1))  # -> [B,repr_dim]

class JEPA(nn.Module):
    def __init__(self, repr_dim: int = 256):
        super().__init__()
        self.encoder   = InceptionEncoder(repr_dim=repr_dim, in_channels=2)
        self.predictor = Predictor(repr_dim=repr_dim, action_dim=2)
        self.repr_dim  = repr_dim

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        states:  [B, T, C, H, W]
        actions: [B, T-1, 2]
        returns: [B, T, repr_dim]
        """
        B, T, C, H, W = states.shape
        preds = []

        # encode initial frame
        s = self.encoder(states[:, 0])  # [B, D]
        preds.append(s)

        # predict each step
        for t in range(actions.shape[1]):  # T-1 steps
            s = self.predictor(s, actions[:, t])
            preds.append(s)

        return torch.stack(preds, dim=1)   # [B, T, D]

    def compute_loss(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        JEPA energy loss: MSE between
        preds[:,1:] (predictions for frames 1..T-1)
        and encoder(states[:,1:]) embeddings.
        """
        # extract dimensions
        B, T, C, H, W = states.shape

        # get all predictions [B, T, D]
        preds = self.forward(states, actions)
        preds_next = preds[:, 1:]             # [B, T-1, D]

        # compute target embeddings for frames 1..T-1
        with torch.no_grad():
            targets = torch.stack(
                [self.encoder(states[:, t]) for t in range(1, T)],
                dim=1
            )  # [B, T-1, D]

        # MSE loss
        return F.mse_loss(preds_next, targets)

def nt_xent_loss(z_i: torch.Tensor, z_j: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """
    NT-Xent contrastive loss between two sets of representations.
    z_i, z_j: [N, D]
    """
    # normalize
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)
    N = z_i.shape[0]
    # cosine similarity matrix
    sim = torch.matmul(z_i, z_j.T) / temperature  # [N,N]
    # positives: diagonal
    pos = torch.exp(torch.diagonal(sim))
    # all similarities
    exp_sim = torch.exp(sim)
    # negative sum for each row
    neg = exp_sim.sum(dim=1) - pos
    # loss for i->j and j->i
    loss_i = -torch.log(pos / (pos + neg)).mean()
    loss_j = -torch.log(pos / (pos + neg)).mean()
    return 0.5 * (loss_i + loss_j)

class MockModel(nn.Module):
    """Dummy model for testing."""
    def __init__(self, device="cuda", output_dim=256):
        super().__init__()
        self.device = device
        self.repr_dim = output_dim

    def forward(self, states, actions):
        B, T, _ = actions.shape
        return torch.randn((B, T + 1, self.repr_dim), device=self.device)


class Prober(nn.Module):
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

        arch_list = list(map(int, arch.split("-"))) if arch else []
        dims = [embedding] + arch_list + [self.output_dim]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU(True))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.prober = nn.Sequential(*layers)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        return self.prober(e)

