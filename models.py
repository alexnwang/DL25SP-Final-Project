from typing import List
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch
from tqdm import tqdm
from copy import deepcopy
from torch.optim.lr_scheduler import CosineAnnealingLR
import timm


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

class ViTEncoder(nn.Module):
    def __init__(self, pretrained: bool = False, embed_dim: int = 256):
        super().__init__()
        self.backbone = timm.create_model('vit_tiny_patch16_224', pretrained=pretrained, in_chans = 2, num_classes = 0)
        self.projector = nn.Linear(self.backbone.num_features, embed_dim)

    def forward(self, x):
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        feats = self.backbone(x)
        emb = self.projector(feats)
        return emb
    
class JEPARecurrentPredictor(nn.Module)：
    def __init__(self, repr_dim = 256, action_dim =2, hidden_dim = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(repr_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, repr_dim)
        )

    def forward(self, z, a):  # z: (B, D), a: (B, 2)
        return self.net(torch.cat([z, a], dim=-1))  # (B, D)

class JEPARecurrent(nn.Module):
    def __init__(self, repr_dim=128):
        super().__init__()
        self.encoder = ViTEncoder(repr_dim)
        self.predictor = JEPARecurrentPredictor(repr_dim)

    def forward(self, states, actions):
        B, T, _ = actions.shape
        z = self.encoder(states[:, 0]) 
        preds = []

        for t in range(T):
            a = actions[:, t]
            z = self.predictor(z, a)  # predict next embedding
            preds.append(z)

        return torch.stack(preds, dim=1)  # (B, T, D)

    @property
    def repr_dim(self):
        return self.encoder.repr_dim
    
    def jepa_loss_mse(self, model, target_encoder, obs_seq, action_seq):
        B, T_plus_1, C, H, W = obs_seq.shape
        T = T_plus_1 - 1
        o0 = obs_seq[:, 0]  # (B, 1, 2, 64, 64)
        pred_embeds = model(o0, action_seq)  # (B, T, D)

        with torch.no_grad():
            target_embeds = torch.stack([
                target_encoder(obs_seq[:, t+1]) for t in range(T)
            ], dim=1)  # (B, T, D)

        loss = F.mse_loss(pred_embeds, target_embeds)
        return loss
    
    def compute_vicreg_loss(self, x, y, sim_coeff=10, var_coeff=25.0, cov_coeff=25.0, eps=1e-4):

        B, D = x.shape

        # Invariance loss
        sim_loss = F.mse_loss(x, y)

        # Variance loss
        def variance_loss(z):
            std = torch.sqrt(z.var(dim=0) + eps)
            return torch.mean(F.relu(1.0 - std))

        var_loss = variance_loss(x) + variance_loss(y)

        # Covariance loss
        def covariance_loss(z):
            z = z - z.mean(dim=0)
            cov = (z.T @ z) / (B - 1)
            off_diag = cov - torch.diag(torch.diag(cov))
            return (off_diag ** 2).sum() / D

        cov_loss = covariance_loss(x) + covariance_loss(y)
        # print(f"[DEBUG] sim_loss: {sim_loss:.4f}, var_loss: {var_loss:.4f}, cov_loss: {cov_loss:.4f}")
        # print(f"sim={sim_loss.item():.2f}, var={var_loss.item():.2f}, cov={cov_loss.item():.2f}")
        total = sim_coeff * sim_loss + var_coeff * var_loss + cov_coeff * cov_loss

        return total, sim_loss.item(), var_loss.item(), cov_loss.item()
    
    def vicreg_loss_recurrent(self, model, target_encoder, obs_seq, action_seq, **kwargs):

        B, T_plus_1, _, _, _ = obs_seq.shape
        T = T_plus_1 - 1

        pred_embeds = model(obs_seq[:, 0:1], action_seq)  # (B, T, D)

        with torch.no_grad():
            target_embeds = torch.stack([
                target_encoder(obs_seq[:, t + 1]) for t in range(T)
            ], dim=1)  # (B, T, D)

        total_loss = 0.0
        total_sim = 0.0
        total_var = 0.0
        total_cov = 0.0
        for t in range(T):
            pred = pred_embeds[:, t]   # (B, D)
            target = target_embeds[:, t]  # (B, D)

            loss_t, sim, var, cov  = self.compute_vicreg_loss(pred, target, **kwargs)
            total_loss += loss_t
            total_sim += sim
            total_var += var
            total_cov += cov
        print(f"[VICREG] sim={total_sim / T:.4f}, var={total_var / T:.4f}, cov={total_cov / T:.4f}")

        return total_loss / T
    
    def update_momentum_encoder(self, encoder, target_encoder, m=0.99):
        for p, p_tgt in zip(encoder.parameters(), target_encoder.parameters()):
            p_tgt.data = m * p_tgt.data + (1.0 - m) * p.data
    
    def train_jepa_recurrent(
        self,
        model,
        train_loader,
        num_epochs=20,
        lr=1e-4,
        device="cuda",
        ema_momentum=0.99,
        log_interval=100,
    ):
        model = model.to(device)
        target_encoder = deepcopy(model.encoder)
        target_encoder.eval()  # EMA target encoder is not trained
        for p in target_encoder.parameters():
            p.requires_grad = False

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0
            steps = 0

            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
                obs_seq = batch.states.to(device)       # (B, T+1, 2, 64, 64)
                action_seq = batch.actions.to(device)   # (B, T, 2)

                loss = self.vicreg_loss_recurrent(model, target_encoder, obs_seq, action_seq)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # EMA update for target encoder
                self.update_momentum_encoder(model.encoder, target_encoder, m=ema_momentum)

                total_loss += loss.item()
                steps += 1

                if batch_idx % log_interval == 0:
                    print(f"[Epoch {epoch+1} | Step {batch_idx}] Loss: {loss.item():.4f}")

            avg_loss = total_loss / steps
            print(f"[Epoch {epoch+1}] Average Loss: {avg_loss:.4f}")

            #check for collapse

            with torch.no_grad():
                pred_z = model(obs_seq[:, 0:1], action_seq)  # (B, T, D)
                flat_z = pred_z.reshape(-1, pred_z.shape[-1])  # → (B×T, D)

            means = flat_z.mean(dim=0)
            stds = flat_z.std(dim=0)

            print(f"[Epoch {epoch+1}] Mean: {means.mean().item():.4f}, "
                f"Std (avg across dims): {stds.mean().item():.4f}, "
                f"Min std: {stds.min().item():.4f}")
            scheduler.step()
        torch.save(model.state_dict(), "model.pt")

    @torch.no_grad()
    def validate_jepa(model, target_encoder, val_loader, loss_fn, device="cuda"):
        model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in val_loader:
            obs_seq = batch.states.to(device)       # (B, T+1, 2, 64, 64)
            action_seq = batch.actions.to(device)   # (B, T, 2)

            loss = loss_fn(model, target_encoder, obs_seq, action_seq)
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"[Validation] Avg Loss: {avg_loss:.4f}")
        return avg_loss

            



