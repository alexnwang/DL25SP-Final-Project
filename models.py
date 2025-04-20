from typing import List
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch
import timm
from loss import VICRegLoss
from utilsls import get_optimizer, get_scheduler 
import math 

vicreg_loss = VICRegLoss()


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
    
# JEPA ARCH 

class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def training_step(self, batch):
        raise NotImplementedError

    def validation_step(self, batch):
        raise NotImplementedError
    
# using pretrained ViT as our encoder 
class DinoV2ViTEncoder(nn.Module):
    """
    ViT-B/14 (DINO-v2) adapted for 2-channel 64x64 inputs.

    Output:  feature map  [B, out_c, 5, 5]
             (because 70x70xpad  ÷ patch-size 14  → 5x5 tokens)
    """
    def __init__(self, config):
        super().__init__()

        # --- load DINO‑v2 ViT‑B/14 backbone ----------------------------
        ckpt = "vit_base_patch14_dinov2.lvd142m"
        self.vit = timm.create_model(
            ckpt, pretrained=True,
            num_classes=0,      # no classifier head
            global_pool='',     # keep token grid
            in_chans=3)         # will convert to 2 below
        # ----------------------------------------------------------------

        # --- convert patch embed conv from 3‑ch to 2‑ch -----------------
        pe_old = self.vit.patch_embed.proj          # Conv2d(3,768,14,14)
        w = pe_old.weight.data                 # [768,3,14,14]

        pe_new = nn.Conv2d(
            in_channels=2, out_channels=w.size(0),
            kernel_size=pe_old.kernel_size,
            stride=pe_old.stride, padding=pe_old.padding, bias=False)

        with torch.no_grad():
            # copy first two RGB filters and rescale to preserve variance
            pe_new.weight[:] = w[:, :2] * (3.0 / 2.0)
        self.vit.patch_embed.proj = pe_new
        # ----------------------------------------------------------------

        self.patch_size   = self.vit.patch_embed.patch_size[0]   # 14
        self.out_c        = config.out_c                        # latent channels
        self.channel_proj = nn.Conv2d(
            in_channels=self.vit.embed_dim,  # 768
            out_channels=self.out_c, kernel_size=1)

    # --------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor  [B*, 2, 64, 64]   (agent + wall)

        Returns
        -------
        feats : Tensor  [B*, out_c, 5, 5]
        """
        # amount of padding needed on H and W to reach multiple of 14
        ph = (self.patch_size - x.size(-2) % self.patch_size) % self.patch_size
        pw = (self.patch_size - x.size(-1) % self.patch_size) % self.patch_size
        # pad bottom/right; order = (left, right, top, bottom)
        x = F.pad(x, (0, pw, 0, ph))               # 64×64 → 70×70

        tokens = self.vit.patch_embed(x)           # [B, 768, 5, 5]
        feats  = self.channel_proj(tokens)         # [B, out_c, 5, 5]
        return feats
    

class ViTPredictor2D(nn.Module):
    """
    Predictor for a 5×5 latent map produced by DinoV2ViTEncoder.

    1. Projects (Δx, Δy) into a 5×5 mask.
    2. Concatenates with current state map  →  channels = (out_c + 1)
    3. Two depth‑wise‑friendly 3×3 convs   →  next‑state map.
    """
    def __init__(self, config):
        super().__init__()
        self.out_c      = config.out_c           # e.g. 1
        self.action_dim = config.action_dim      # 2
        self.side       = 5                      # 70÷14

        # (1) action → mask
        self.action_proj = nn.Linear(self.action_dim,
                                     self.side * self.side)

        # (2) φ(state, action)  →  next‑state
        hidden_c = max(8, self.out_c * 4)        # tiny but expressive
        self.net = nn.Sequential(
            nn.Conv2d(self.out_c + 1, hidden_c, 3, padding=1, groups=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_c, self.out_c, 3, padding=1))

        # small weight init so predictor starts close to identity
        for m in self.net:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, s_map, action):
        """
        Parameters
        ----------
        s_map  : [B, out_c, 5, 5]   - current latent map
        action : [B, 2]             - (DeltaX, DeltaY) in grid units

        Returns
        -------
        next_s : [B, out_c, 5, 5]   - predicted next latent map
        """
        B = action.size(0)
        a_mask = (self.action_proj(action)          # [B, 25]
                  .view(B, 1, self.side, self.side))# [B, 1, 5, 5]

        x = torch.cat([s_map, a_mask], dim=1)       # [B, out_c+1, 5, 5]
        next_s = self.net(x)                        # [B, out_c, 5, 5]
        return next_s
class ActionRegularizer2D(nn.Module):
    def __init__(self, config, embed_dim, action_dim):
        super().__init__()
        self.output_side = int(
            math.sqrt(embed_dim)
        )  # Calculate the side of the 2D embedding
        self.action_reg_net = nn.Sequential(
            nn.Conv2d(config.out_c, 16, kernel_size=3, padding=1),  # 2D conv layer
            nn.ReLU(),
            nn.Conv2d(16, config.out_c, kernel_size=3, padding=1),  # Output C' channels
            nn.Flatten(),  # Flatten to prepare for linear mapping
            nn.Linear(
                self.output_side * self.output_side * config.out_c, action_dim
            ),  # Map to action_dim
        )

    def forward(self, states_embed, pred_states):
        """
        Args:
            states_embed: Tensor of shape (B*(T-1), C', output_side, output_side) - previous state embeddings
            pred_states: Tensor of shape (B*(T-1), C', output_side, output_side) - predicted state embeddings

        Returns:
            predicted_actions: Tensor of shape (B*(T-1), action_dim) - predicted actions
        """
        # Calculate embedding differences
        embedding_diff = (
            pred_states - states_embed
        )  # (B*(T-1), C', output_side, output_side)

        # Predict actions from embedding differences
        predicted_actions = self.action_reg_net(embedding_diff)  # (B*(T-1), action_dim)
        return predicted_actions


class ActionRegularizationJEPA2D_ViT(BaseModel):
    """
    JEPA with
      • DINO-v2 ViT-B/14 encoder      → latent [B, out_c, 5, 5]
      • small 3x3-conv predictor
      • action-regulariser   (predict Δx,Δy from ẑ - z)
      • VICReg invariance / variance / covariance loss
    """

    # ---------- ctor ---------------------------------------------------
    def __init__(self, config):
        super().__init__(config)

        # backbone pieces ------------------------------------------------
        self.enc  = DinoV2ViTEncoder(config)      # outputs [B, C=out_c, 5, 5]
        self.pred = ViTPredictor2D(config)        # maps (state, action) → next_state
        self.action_reg_net = ActionRegularizer2D(
            config, embed_dim=config.out_c, action_dim=config.action_dim)

        # optim / sched --------------------------------------------------
        self.optimizer = get_optimizer(config, self.parameters())
        self.scheduler = get_scheduler(self.optimizer, config)

        # consts ---------------------------------------------------------
        self.C  = config.out_c              # latent channels (usually 1)
        self.HW = 5 * 5                     # flattened spatial size
        self.cfg = config

    # ---------- forward -------------------------------------------------
    def forward(self, states, actions, teacher_forcing=True):
        """
        states  : [B, T, 2, 64, 64]
        actions : [B, T-1, 2]
        """
        B, T, _, _, _ = states.shape
        # (1) encode all frames in one batch → [B*T, C, 5, 5]
        z = self.enc(states.view(-1, 2, 64, 64))
        z = z.view(B, T, self.C, 5, 5)          # (B,T,C,5,5)

        if teacher_forcing:                     # single‑shot predictor
            s_in  = z[:, :-1]                  # (B,T‑1,C,5,5)
            a_in  = actions                     # (B,T‑1,2)
            y_hat = self.pred(
                s_in.reshape(-1, self.C, 5, 5),
                a_in.reshape(-1, self.cfg.action_dim)
            ).view(B, T-1, self.C, 5, 5)

            # prepend the true first state
            preds = torch.cat([z[:, :1], y_hat], dim=1)   # (B,T,C,5,5)
            return preds, z

        # ---------- recurrent roll‑out ---------------------
        preds = [z[:, 0]]                                   # list of (B,C,5,5)
        for t in range(1, T):
            p = self.pred(preds[-1], actions[:, t-1])
            preds.append(p)
        preds = torch.stack(preds, dim=1)                   # (B,T,C,5,5)
        return preds, z

    # ---------- loss helpers -------------------------------------------
    @staticmethod
    def _flat4d(x):            # (B,T,C,H,W) → (B*T, C*H*W)
        B, T, C, H, W = x.shape
        return x.reshape(B*T, C*H*W)

    def _vicreg(self, preds, targets):
        loss, inv, var, cov = VICRegLoss()(
            self._flat4d(preds[:,1:]), self._flat4d(targets[:,1:]),
            self.cfg.vicreg_loss.lambda_invariance,
            self.cfg.vicreg_loss.mu_variance,
            self.cfg.vicreg_loss.nu_covariance)
        return loss, inv, var, cov

    def _action_reg(self, s_prev, s_next, act):
        act_hat = self.action_reg_net(s_prev, s_next)
        return F.mse_loss(act_hat, act)

    # ---------- training / val step ------------------------------------
    def training_step(self, batch, device):
        states  = batch.states.to(device, non_blocking=True)
        actions = batch.actions.to(device, non_blocking=True)

        preds, z_true = self(states, actions, teacher_forcing=True)

        # losses ---------------------------------------------------------
        vic, inv, var, cov = self._vicreg(preds, z_true)

        B,T = actions.shape[:2]
        act_reg = self._action_reg(
            z_true[:, :-1].reshape(-1, self.C, 5, 5),
            preds[:, 1:].reshape(-1, self.C, 5, 5),
            actions.reshape(-1, self.cfg.action_dim))

        loss = vic + self.cfg.lambda_reg * act_reg

        # optimisation ---------------------------------------------------
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step(); self.scheduler.step()

        # logging dict ---------------------------------------------------
        return {
            "loss": loss.item(),
            "vicreg": vic.item(),
            "action_reg": act_reg.item(),
            "inv": inv.item(), "var": var.item(), "cov": cov.item(),
            "lr": self.optimizer.param_groups[0]["lr"]
        }

    @torch.no_grad()
    def validation_step(self, batch):
        preds, z = self(batch.states, batch.actions, teacher_forcing=True)
        mse = F.mse_loss(preds[:,1:], z[:,1:])
        return {"loss": mse.item()}
