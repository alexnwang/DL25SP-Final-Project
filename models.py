# models.py

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# ----------------------------------------------------------------------------
# 1) MockModel for main.py compatibility
# ----------------------------------------------------------------------------
class MockModel(nn.Module):
    """
    Does nothing. Just for testing.
    """
    def __init__(self, device="cuda", output_dim=256):
        super().__init__()
        self.device = torch.device(device)
        self.repr_dim = output_dim

    def forward(self, states, actions):
        B, T, _ = actions.shape
        # return random embeddings [B, T, repr_dim]
        return torch.randn((B, T, self.repr_dim), device=self.device)

# ----------------------------------------------------------------------------
# 2) Utility MLP builder for Prober & elsewhere
# ----------------------------------------------------------------------------
def build_mlp(layers_dims: List[int]) -> nn.Sequential:
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i+1]))
        layers.append(nn.BatchNorm1d(layers_dims[i+1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)

# ----------------------------------------------------------------------------
# 3) NT-Xent contrastive loss
# ----------------------------------------------------------------------------
def nt_xent_loss(z_i: torch.Tensor, z_j: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)
    N = z_i.size(0)
    sim = torch.matmul(z_i, z_j.t()) / temperature  # [N,N]
    exp_sim = torch.exp(sim)
    pos = torch.diagonal(exp_sim)                   # [N]
    neg = exp_sim.sum(dim=1) - pos                  # [N]
    loss = -torch.log(pos / (pos + neg)).mean()
    return loss

# ----------------------------------------------------------------------------
# 4) Variational Inception-V3 encoder (random init)
# ----------------------------------------------------------------------------
class VEncoder(nn.Module):
    """
    Inception-V3 backbone → (µ, logσ²) → z via reparameterization.
    Upsamples inputs to 128×128 instead of 299×299.
    """
    def __init__(self, repr_dim: int = 256, in_channels: int = 2):
        super().__init__()
        inc = models.inception_v3(
            weights=None,
            aux_logits=False,
            transform_input=False
        )
        # adapt first conv for 2-channel input
        inc.Conv2d_1a_3x3 = nn.Conv2d(
            in_channels, 32,
            kernel_size=3, stride=2, padding=1
        )

        # keep full Inception stack
        self.features = nn.Sequential(
            inc.Conv2d_1a_3x3,
            inc.Conv2d_2a_3x3,
            inc.Conv2d_2b_3x3,
            nn.MaxPool2d(3, 2),
            inc.Conv2d_3b_1x1,
            inc.Conv2d_4a_3x3,
            nn.MaxPool2d(3, 2),
            inc.Mixed_5b,
            inc.Mixed_5c,
            inc.Mixed_5d,
            inc.Mixed_6a,
            inc.Mixed_6b,
            inc.Mixed_6c,
            inc.Mixed_6d,
            inc.Mixed_6e,
            inc.Mixed_7a,
            inc.Mixed_7b,
            inc.Mixed_7c,
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten()      # → [B, 2048]
        )
        self.fc_mu     = nn.Linear(2048, repr_dim)
        self.fc_logvar = nn.Linear(2048, repr_dim)

    def forward(self, x: torch.Tensor):
        # x: [B,2,65,65] → upsample → [B,2,128,128]
        x = F.interpolate(
            x, size=(128,128),
            mode='bilinear', align_corners=False
        )
        h = self.features(x)            # [B,2048]
        mu     = self.fc_mu(h)          # [B,repr_dim]
        logvar = self.fc_logvar(h)      # [B,repr_dim]
        std  = (0.5 * logvar).exp()
        eps  = torch.randn_like(std)
        z    = mu + eps * std           # reparameterization
        return z, mu, logvar

# ----------------------------------------------------------------------------
# 5) Contrastive projection head
# ----------------------------------------------------------------------------
class ProjectionHead(nn.Module):
    def __init__(self, repr_dim: int = 256, proj_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(repr_dim, repr_dim),
            nn.ReLU(True),
            nn.Linear(repr_dim, proj_dim),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ----------------------------------------------------------------------------
# 6) JEPA predictor with residual skip
# ----------------------------------------------------------------------------
class Predictor(nn.Module):
    def __init__(self, repr_dim: int = 256, action_dim: int = 2):
        super().__init__()
        self.delta = nn.Sequential(
            nn.Linear(repr_dim + action_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, repr_dim),
        )
    def forward(self, h: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        d = self.delta(torch.cat([h, a], dim=1))
        return h + d  # skip connection

# ----------------------------------------------------------------------------
# 7) Full JEPA model
# ----------------------------------------------------------------------------
class JEPA(nn.Module):
    def __init__(self, repr_dim: int = 256, proj_dim: int = 256):
        super().__init__()
        self.encoder   = VEncoder(repr_dim=repr_dim, in_channels=2)
        self.predictor = Predictor(repr_dim=repr_dim, action_dim=2)
        self.proj_head = ProjectionHead(repr_dim=repr_dim, proj_dim=proj_dim)
        self.repr_dim  = repr_dim

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            states:  [B, T, 2, H, W]
            actions: [B, T-1, 2]
        Returns:
            preds:   [B, T, repr_dim]  (incl. initial state)
        """
        B, T, C, H, W = states.shape
        preds = []
        # 1) encode initial frame
        h, _, _ = self.encoder(states[:, 0])
        preds.append(h)
        # 2) predict residual steps
        for t in range(actions.shape[1]):
            h = self.predictor(h, actions[:, t])  # [B,repr_dim]
            preds.append(h)
        return torch.stack(preds, dim=1)  # [B, T, repr_dim]

    def compute_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        aug_states: torch.Tensor,
        model_m: "JEPA",
        alpha: float = 1.0,
        beta: float = 0.1,
        temperature: float = 0.5
    ):
        """
        Returns the three losses:
          - jepa_loss (predictive MSE)
          - kl_loss   (variational KL)
          - ctr_loss  (contrastive NT-Xent)
        """
        B, T, C, H, W = states.shape

        # — JEPA energy loss —
        all_preds = self.forward(states, actions)   # [B, T, D]
        preds_next = all_preds[:, 1:]               # [B, T-1, D]
        with torch.no_grad():
            targets = torch.stack([
                self.encoder(states[:, t])[1]       # mu of each frame
                for t in range(1, T)
            ], dim=1)                               # [B, T-1, D]
        jepa_loss = F.mse_loss(preds_next, targets)

        # — KL divergence —
        flat = states.view(-1, C, H, W)             # [B*T, 2, H, W]
        _, mu_all, logvar_all = self.encoder(flat)
        kl_terms = -0.5 * (1 + logvar_all - mu_all.pow(2) - logvar_all.exp())
        kl_loss = kl_terms.sum(dim=1).mean()

        # — Contrastive (MoCo-style) —
        z_o, _, _    = self.encoder(flat)           # [B*T, D]
        proj_o       = self.proj_head(z_o)

        flat_aug     = aug_states.view(-1, C, H, W)
        with torch.no_grad():
            z_j, _, _= model_m.encoder(flat_aug)
            proj_j   = model_m.proj_head(z_j)

        ctr_loss = nt_xent_loss(proj_o, proj_j, temperature)

        return jepa_loss, kl_loss, ctr_loss

# ----------------------------------------------------------------------------
# 8) Momentum-encoder helpers
# ----------------------------------------------------------------------------
def init_momentum_params(model_q: nn.Module, model_k: nn.Module):
    """Copy parameters from online (q) to momentum (k)."""
    for pq, pk in zip(model_q.parameters(), model_k.parameters()):
        pk.data.copy_(pq.data)
        pk.requires_grad = False

def update_momentum_params(model_q: nn.Module, model_k: nn.Module, m: float):
    """EMA update: θ_k ← m θ_k + (1−m) θ_q"""
    for pq, pk in zip(model_q.parameters(), model_k.parameters()):
        pk.data.mul_(m).add_(pq.data, alpha=1-m)

# ----------------------------------------------------------------------------
# 9) Prober for downstream coordinate extraction
# ----------------------------------------------------------------------------
class Prober(nn.Module):
    def __init__(
        self,
        embedding: int,
        arch: str,
        output_shape: List[int]
    ):
        super().__init__()
        self.output_dim   = int(torch.prod(torch.tensor(output_shape)))
        arch_list = list(map(int, arch.split("-"))) if arch else []
        f = [embedding] + arch_list + [self.output_dim]
        layers = []
        for i in range(len(f) - 2):
            layers.append(nn.Linear(f[i], f[i+1]))
            layers.append(nn.ReLU(True))
        layers.append(nn.Linear(f[-2], f[-1]))
        self.prober = nn.Sequential(*layers)
        self.output_shape = output_shape

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        return self.prober(e).view(-1, *self.output_shape)

