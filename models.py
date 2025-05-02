from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# ----------------------------------------------------------------------------
# 1) MockModel for main.py compatibility
# ----------------------------------------------------------------------------
class MockModel(nn.Module):
    def __init__(self, device="cuda", output_dim=256):
        super().__init__()
        self.device = torch.device(device)
        self.repr_dim = output_dim

    def forward(self, states, actions):
        B, T, _ = actions.shape
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
    sim = torch.matmul(z_i, z_j.t()) / temperature
    exp_sim = torch.exp(sim)
    pos = torch.diagonal(exp_sim)
    neg = exp_sim.sum(dim=1) - pos
    loss = -torch.log(pos / (pos + neg)).mean()
    return loss

# ----------------------------------------------------------------------------
# 4) Variational Inception-V3 encoder (random init)
# ----------------------------------------------------------------------------
class VEncoder(nn.Module):
    def __init__(self, repr_dim: int = 256, in_channels: int = 2):
        super().__init__()
        inc = models.inception_v3(weights=None, aux_logits=False, transform_input=False)
        inc.Conv2d_1a_3x3 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.features = nn.Sequential(
            inc.Conv2d_1a_3x3, inc.Conv2d_2a_3x3, inc.Conv2d_2b_3x3,
            nn.MaxPool2d(3, 2), inc.Conv2d_3b_1x1, inc.Conv2d_4a_3x3,
            nn.MaxPool2d(3, 2), inc.Mixed_5b, inc.Mixed_5c, inc.Mixed_5d,
            inc.Mixed_6a, inc.Mixed_6b, inc.Mixed_6c, inc.Mixed_6d, inc.Mixed_6e,
            inc.Mixed_7a, inc.Mixed_7b, inc.Mixed_7c,
            nn.AdaptiveAvgPool2d((1,1)), nn.Flatten()
        )
        self.fc_mu = nn.Linear(2048, repr_dim)
        self.fc_logvar = nn.Linear(2048, repr_dim)

    def forward(self, x: torch.Tensor):
        x = F.interpolate(x, size=(128,128), mode='bilinear', align_corners=False)
        h = self.features(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar

# ----------------------------------------------------------------------------
# 5) Contrastive projection head
# ----------------------------------------------------------------------------
class ProjectionHead(nn.Module):
    def __init__(self, repr_dim: int = 256, proj_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(repr_dim, repr_dim), nn.ReLU(True), nn.Linear(repr_dim, proj_dim),
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
            nn.Linear(repr_dim + action_dim, 512), nn.ReLU(True), nn.Linear(512, repr_dim),
        )
    def forward(self, h: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        d = self.delta(torch.cat([h, a], dim=1))
        return h + d

# ----------------------------------------------------------------------------
# 7) Full JEPA model
# ----------------------------------------------------------------------------
class JEPA(nn.Module):
    def __init__(self, repr_dim: int = 256, proj_dim: int = 256):
        super().__init__()
        self.encoder = VEncoder(repr_dim=repr_dim, in_channels=2)
        self.predictor = Predictor(repr_dim=repr_dim, action_dim=2)
        self.proj_head = ProjectionHead(repr_dim=repr_dim, proj_dim=proj_dim)
        self.repr_dim = repr_dim

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = states.shape
        preds = []
        h, _, _ = self.encoder(states[:, 0])
        preds.append(h)
        for t in range(actions.shape[1]):
            h = self.predictor(h, actions[:, t])
            preds.append(h)
        return torch.stack(preds, dim=1)

    def compute_loss(self, states, actions, aug_states, model_m, alpha=1.0, beta=0.1, temperature=0.5):
        B, T, C, H, W = states.shape
        all_preds = self.forward(states, actions)
        preds_next = all_preds[:, 1:]
        with torch.no_grad():
            targets = torch.stack([self.encoder(states[:, t])[1] for t in range(1, T)], dim=1)
        jepa_loss = F.mse_loss(preds_next, targets)

        flat = states.view(-1, C, H, W)
        _, mu_all, logvar_all = self.encoder(flat)
        kl_terms = -0.5 * (1 + logvar_all - mu_all.pow(2) - logvar_all.exp())
        kl_loss = kl_terms.sum(dim=1).mean()

        z_o, _, _ = self.encoder(flat)
        proj_o = self.proj_head(z_o)
        flat_aug = aug_states.view(-1, C, H, W)
        with torch.no_grad():
            z_j, _, _ = model_m.encoder(flat_aug)
            proj_j = model_m.proj_head(z_j)
        ctr_loss = nt_xent_loss(proj_o, proj_j, temperature)

        return jepa_loss, kl_loss, ctr_loss

# ----------------------------------------------------------------------------
# 8) Momentum-encoder helpers
# ----------------------------------------------------------------------------
def init_momentum_params(model_q: nn.Module, model_k: nn.Module):
    for pq, pk in zip(model_q.parameters(), model_k.parameters()):
        pk.data.copy_(pq.data)
        pk.requires_grad = False

def update_momentum_params(model_q: nn.Module, model_k: nn.Module, m: float):
    for pq, pk in zip(model_q.parameters(), model_k.parameters()):
        pk.data.mul_(m).add_(pq.data, alpha=1 - m)

# ----------------------------------------------------------------------------
# 9) Prober for downstream coordinate extraction
# ----------------------------------------------------------------------------
class Prober(nn.Module):
    def __init__(self, embedding: int, arch: str, output_shape: List[int]):
        super().__init__()
        self.output_dim = int(torch.prod(torch.tensor(output_shape)))
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
