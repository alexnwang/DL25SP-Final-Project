#!/usr/bin/env python3
"""
train_jepa.py

Train a simple JEPA model using DINO-V2 patch embeddings and a recurrent MLP predictor.
No proprio or decoder networks are used. The model is saved to a checkpoint for later evaluation.
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from einops import rearrange, repeat
from torch.optim.lr_scheduler import LambdaLR
import json
import datetime
import subprocess
import matplotlib.pyplot as plt
from typing import Optional
# configure wandb directories before importing wandb to avoid permission issues
os.environ['WANDB_DIR'] = '/gpfs/scratch/wz1492/DL25SP-Final-Project/wandb_tmp'
os.environ['WANDB_CONFIG_DIR'] = os.environ['WANDB_DIR']
os.environ['XDG_CONFIG_HOME'] = os.environ['WANDB_DIR']
# create directories
os.makedirs(os.environ['WANDB_DIR'], exist_ok=True)
os.makedirs(os.environ['WANDB_CONFIG_DIR'], exist_ok=True)
from pipeline import CheckpointEvalPipeline
import wandb

# helper functions for learning rate schedule and latent normalization
def linear_warmup_decay(warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
    return lr_lambda

def normalize_latents(z):
    z = z - z.mean(dim=0, keepdim=True)
    z = z / (z.std(dim=0, keepdim=True) + 1e-6)
    return z

# ============================================================================
# Encoder: DINO-V2 patch embeddings
# ============================================================================
class DinoV2Encoder(nn.Module):
    def __init__(self, name: str, feature_key: str):
        super().__init__()
        # disable Hub fork check
        import torch.hub
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        # load pretrained DINOv2 model
        self.base_model = torch.hub.load("facebookresearch/dinov2", name)
        self.feature_key = feature_key
        self.emb_dim = self.base_model.num_features
        if feature_key == "x_norm_patchtokens":
            self.latent_ndim = 2
        elif feature_key == "x_norm_clstoken":
            self.latent_ndim = 1
        else:
            raise ValueError(f"Invalid feature_key: {feature_key}")
        self.patch_size = self.base_model.patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W)
        # ensure height and width are multiples of patch size
        B, C, H, W = x.shape
        ps = self.patch_size
        # compute next multiples of patch size
        new_H = ((H + ps - 1) // ps) * ps
        new_W = ((W + ps - 1) // ps) * ps
        if new_H != H or new_W != W:
            x = F.interpolate(x, size=(new_H, new_W), mode='bilinear', align_corners=False)
        feats = self.base_model.forward_features(x)
        emb = feats[self.feature_key]
        # if collapsed to cls token, add patch dim
        if self.latent_ndim == 1:
            emb = emb.unsqueeze(1)
        return emb  # (B, num_patches, emb_dim)

# Predictor: ViT-based transformer consuming patch sequences across history
NUM_FRAMES = None  # will be set in ViTPredictor.__init__
NUM_PATCHES = None

def generate_mask_matrix(npatch, nwindow):
    zeros = torch.zeros(npatch, npatch)
    ones = torch.ones(npatch, npatch)
    rows = []
    for i in range(nwindow):
        row = torch.cat([ones] * (i+1) + [zeros] * (nwindow - i-1), dim=1)
        rows.append(row)
    mask = torch.cat(rows, dim=0).unsqueeze(0).unsqueeze(0)
    return mask

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
    def forward(self, x):
        # use flash-scaled dot product attention when available
        x_norm = self.norm(x)
        qkv = self.to_qkv(x_norm).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        try:
            # PyTorch 2.0+ flash attention
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=True
            )
        except AttributeError:
            # fallback to manual attention
            b, n, _ = x.size()
            mask = generate_mask_matrix(NUM_PATCHES, NUM_FRAMES).to(x.device)
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            dots = dots.masked_fill(mask[:, :, :n, :n] == 0, float('-inf'))
            attn = self.attend(dots)
            attn = F.dropout(attn, self.dropout.p, self.training)
            attn_out = torch.matmul(attn, v)
        out = rearrange(attn_out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        global NUM_FRAMES, NUM_PATCHES
        self.norm = nn.LayerNorm(dim)
        # store transformer blocks as a Python list of (attn, ff) tuples
        self.layers = []
        for i in range(depth):
            attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
            ff = FeedForward(dim, mlp_dim, dropout=dropout)
            # register submodules so parameters are tracked
            self.add_module(f"attn_{i}", attn)
            self.add_module(f"ff_{i}", ff)
            self.layers.append((attn, ff))
    def forward(self, x):
        # sequentially apply each attention+feedforward block
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class ViTPredictor(nn.Module):
    def __init__(self, num_patches, num_frames, dim, depth=6, heads=16, dim_head=64, mlp_dim=2048, dropout=0.1, emb_dropout=0., pool='mean'):
        super().__init__()
        global NUM_FRAMES, NUM_PATCHES
        NUM_FRAMES = num_frames
        NUM_PATCHES = num_patches
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames * num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
    def forward(self, x):
        # x: (b, num_frames*num_patches, dim)
        x = x + self.pos_embedding[:, :x.size(1)]
        x = self.dropout(x)
        x = self.transformer(x)
        return x

# ============================================================================
# JEPA wrapper: freeze encoder, train predictor
# ============================================================================
class JEPAWrapper(nn.Module):
    def __init__(
        self,
        encoder_name: str = "dinov2_vits14",
        feature_key: str = "x_norm_patchtokens",
        action_emb_dim: int = 10,
        num_hist: int = 3,
        pred_depth: int = 6,
        pred_heads: int = 16,
        pred_dim_head: int = 64,
        pred_mlp_dim: int = 2048,
        pred_dropout: float = 0.1,
        pred_emb_dropout: float = 0.0,
        pred_pool: str = 'mean',
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # patch-embedding encoder
        self.encoder = DinoV2Encoder(encoder_name, feature_key).to(self.device)
        # freeze encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()
        # record representation dimension
        self.repr_dim = self.encoder.emb_dim
        # action encoder: temporal Conv1d network mapping 2D action sequences to repr_dim per frame
        self.action_emb_dim = action_emb_dim
        self.action_encoder = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=self.action_emb_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(pred_emb_dropout),
            nn.Conv1d(in_channels=self.action_emb_dim, out_channels=self.repr_dim, kernel_size=3, padding=1),
            nn.Dropout(pred_emb_dropout),
        ).to(self.device)
        # register ImageNet normalization on the correct device
        self.register_buffer('rgb_mean', torch.tensor([0.485,0.456,0.406], device=self.device).view(1,3,1,1))
        self.register_buffer('rgb_std',  torch.tensor([0.229,0.224,0.225], device=self.device).view(1,3,1,1))
        # compute normalized patch coordinates for each patch token
        grid_size = 224 // self.encoder.patch_size
        xs = torch.linspace(0, 1, grid_size, device=self.device)
        ys = torch.linspace(0, 1, grid_size, device=self.device)
        yy, xx = torch.meshgrid(ys, xs)
        coords = torch.stack([xx, yy], dim=-1).view(-1, 2)  # (P,2)
        self.register_buffer('patch_coords', coords)
        # action token coordinate indicator
        action_coord = torch.tensor([-1.0, -1.0], device=self.device)
        self.register_buffer('action_coords', action_coord)
        # combined embedding dimension (DINO-V2 feat dim + coord dim)
        self.embed_dim = self.repr_dim + coords.size(-1)
        # compute number of patches per frame (assume 224x224 input)
        num_side = 224 // self.encoder.patch_size
        num_patches = num_side * num_side
        self.num_patches = num_patches
        extended_patches = num_patches + 1
        self.predictor = ViTPredictor(
            num_patches=extended_patches,
            num_frames=num_hist,
            dim=self.embed_dim,
            depth=pred_depth,
            heads=pred_heads,
            dim_head=pred_dim_head,
            mlp_dim=pred_mlp_dim,
            dropout=pred_dropout,
            emb_dropout=pred_emb_dropout,
            pool=pred_pool,
        ).to(self.device)
        self.num_hist = num_hist
        # initialize attention pooling head if using attention pooling
        if pred_pool == 'attn':
            self.attn_pool = nn.Linear(self.embed_dim, 1).to(self.device)

    def forward(self, states: torch.Tensor, actions: torch.Tensor):
        """
        states : (B, T, 3, H, W)
        actions: (B, T-1, 2)
        returns : (B, T, embed_dim)  – per-frame pooled representation
        """
        B, T, C, H, W = states.shape
        device = self.device

        x = states.view(-1, C, H, W)
        if C < 3:
            x = x.repeat(1, 3//C + 1, 1, 1)[:, :3]          # ensure RGB

        x = F.interpolate(x, 224, mode='bilinear', align_corners=False)
        x = (x - self.rgb_mean) / self.rgb_std

        with torch.no_grad():
            patches = self.encoder(x)                       # (B*T,P,repr_dim)

        P = self.num_patches
        patches = patches.view(B, T, P, self.repr_dim)
        coords  = self.patch_coords.view(1, 1, P, 2)
        patches = torch.cat([patches, coords.expand_as(patches[...,:1])], dim=-1)  # → embed_dim

        # -------- action token ---------------------------------------------------
        # embed actions per timestep using temporal Conv1d encoder
        # actions: (B, T-1, 2) -> permute to (B,2,T-1)
        a_ts = actions.permute(0, 2, 1)  # (B, 2, T-1)
        a_emb_ts = self.action_encoder(a_ts)  # (B, repr_dim, T-1)
        # pad to match T timesteps
        pad = torch.zeros(B, self.repr_dim, 1, device=device)  # (B, repr_dim, 1)
        a_emb = torch.cat([a_emb_ts, pad], dim=2)  # (B, repr_dim, T)
        # add action token coordinate dims
        coord_act = self.action_coords.view(1, 2, 1).expand(B, 2, T)  # (B,2,T)
        a_emb = torch.cat([a_emb, coord_act], dim=1)  # (B, embed_dim, T)
        a_tok = a_emb.permute(0, 2, 1).unsqueeze(2)  # (B, T, 1, embed_dim)

        tokens = torch.cat([patches, a_tok], dim=2)         # (B,T,P+1,embed_dim)
        inp    = rearrange(tokens, 'b t p d -> b (t p) d')

        out = self.predictor(inp)
        out = out.view(B, T, P+1, self.embed_dim)[:, :, :P]  # keep visual tokens
        # apply pooling strategy across patches
        if self.predictor.pool == 'mean':
            rep = out.mean(dim=2)
        elif self.predictor.pool == 'cls':
            rep = out[:, :, 0, :]
        elif self.predictor.pool == 'attn':
            logits = self.attn_pool(out)                    # (B,T,P,1)
            weights = F.softmax(logits.squeeze(-1), dim=2)   # (B,T,P)
            rep = torch.sum(out * weights.unsqueeze(-1), dim=2)  # (B,T,embed_dim)
        else:
            raise ValueError(f"Invalid pooling strategy: {self.predictor.pool}")
        return rep

    # --- helper --------------------------------------------------------------
    def _encode_frames(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 3, H, W)  – already resized to 224×224 & RGB
        returns (B, num_patches, embed_dim) with coords concatenated
        """
        with torch.no_grad():
            z = self.encoder(x)                                    # (B,P,D0)
        coords = self.patch_coords.view(1, self.num_patches, 2)    # (1,P,2)
        coords = coords.expand(z.size(0), -1, -1)
        return torch.cat([z, coords], dim=-1)                      # (B,P,D)

    # --- NEW -----------------------------------------------------------------
    def rollout(self,
        init_frame: torch.Tensor,      # (B,1,3,H,W)
        actions:    torch.Tensor,      # (B,T-1,2)
        truth_embeddings: Optional[torch.Tensor] = None,  # (B,T,num_patches,embed_dim)
        sched_sample_prob: float = 1.0) -> torch.Tensor:  # probability of using model's prediction instead of ground truth
        """
        Autoregressively predicts embeddings for all future steps with optional scheduled sampling.
        Returns: (B, T, num_patches, embed_dim) – the first slice (t=0)
                 is just the encoded init_frame.
        """
        B, _, C, H, W = init_frame.shape
        Tm1 = actions.size(1)                   # T-1
        device = self.device

        # -- encode t=0 --------------------------------------------------------
        x0 = init_frame[:,0]  # (B,C,H,W)
        # ensure 3 channels
        if x0.shape[1] == 2:
            x0 = torch.cat([x0, x0[:,0:1]], dim=1)
        elif x0.shape[1] == 1:
            x0 = x0.repeat(1,3,1,1)
        frame0 = F.interpolate(x0, size=(224,224), mode='bilinear', align_corners=False)
        frame0 = (frame0 - self.rgb_mean) / self.rgb_std           # normalise
        latent_t = self._encode_frames(frame0)                     # (B,P,D)
        latents  = [latent_t]                                      # list of (B,P,D)

        # -- step through actions with scheduled sampling --------------------
        for step in range(Tm1):
            # Scheduled sampling: decide whether to use predicted latent or ground truth
            if truth_embeddings is not None:
                truth_latent = truth_embeddings[:, step]  # (B,num_patches,embed_dim)
                if sched_sample_prob < 1.0:
                    rand = torch.rand(B, device=device)
                    mask = (rand < sched_sample_prob).view(B,1,1)  # (B,1,1)
                    latent_input = latent_t * mask + truth_latent * (~mask)
                else:
                    latent_input = latent_t
            else:
                latent_input = latent_t

            # embed current action token using temporal Conv1d encoder
            a_ts = actions[:, step:step+1].permute(0, 2, 1)  # (B,2,1)
            a_emb_ts = self.action_encoder(a_ts)  # (B, repr_dim, 1)
            # add action token coordinate dims
            coord_act = self.action_coords.view(1, 2, 1).expand(B, 2, 1)  # (B,2,1)
            a_emb = torch.cat([a_emb_ts, coord_act], dim=1)  # (B, embed_dim, 1)
            a_emb = a_emb.permute(0, 2, 1).unsqueeze(2)  # (B, 1, 1, embed_dim)

            # 2 concatenate "current latent patches" with that action token
            token_in = torch.cat([latent_input.unsqueeze(1), a_emb], dim=2)  # (B,1,num_patches+1,embed_dim)
            flat_in  = rearrange(token_in, 'b t p d -> b (t p) d')           # (B,(num_patches+1),embed_dim)

            # 3 one transformer pass (causal mask inside Attention keeps order)
            out = self.predictor(flat_in)                                   # (B,(num_patches+1),embed_dim)
            out = out.view(B, 1, self.num_patches+1, self.embed_dim)        # (B,1,num_patches+1,embed_dim)
            latent_t = out[:,0,:self.num_patches]                           # (B,num_patches,embed_dim)

            latents.append(latent_t)

        return torch.stack(latents, dim=1)     # (B, T, P, D)

# ============================================================================
# Dataset: sliding windows of states/actions
# ============================================================================
class JEPATrainDataset(Dataset):
    def __init__(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        num_hist: int,
        num_pred: int,
        transform=None,
    ):
        """
        states: (N, T, C, H, W)
        actions: (N, T-1, 2)
        Creates sliding windows of length num_hist+num_pred
        """
        self.states = states
        self.actions = actions
        self.num_hist = num_hist
        self.num_pred = num_pred
        self.window = num_hist + num_pred
        self.N, self.T, C, H, W = states.shape
        self.transform = transform

    def __len__(self):
        # number of windows per trajectory
        if self.T < self.window:
            return 0
        return self.N * (self.T - self.window + 1)

    def __getitem__(self, idx):
        # map linear idx -> traj_idx, start
        per_traj = self.T - self.window + 1
        traj_idx = idx // per_traj
        start = idx % per_traj
        end = start + self.window
        # load window
        frames = self.states[traj_idx, start:end].copy()  # (W, C, H, W)
        acts = self.actions[traj_idx, start:end - 1].copy()  # (W-1, 2)
        # to torch and scale
        frames = torch.from_numpy(frames).float() / 255.0
        # build 3-channel images: first two channels + duplicate channel0
        _, C, H, W = frames.shape
        imgs = torch.zeros((self.window, 3, H, W), dtype=torch.float32)
        imgs[:, :2] = frames
        imgs[:, 2] = frames[:, 0]
        # apply transform per frame
        if self.transform:
            out = []
            for img in imgs:
                # img is (3,H,W)
                out.append(self.transform(img))
            imgs = torch.stack(out)
        return imgs, torch.from_numpy(acts).float()

# ============================================================================
# Main training loop
# ============================================================================
def main():
    parser = argparse.ArgumentParser("Train JEPA with DINO-V2 embeddings")
    parser.add_argument(
        "--data-dir", type=str, default="/gpfs/scratch/wz1492/data/train"
    )
    parser.add_argument("--epochs", type=int, default=1, help="number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=32, help="training batch size")
    parser.add_argument("--num-hist", type=int, default=16, help="history frames for JEPA prediction")
    parser.add_argument("--num-pred", type=int, default=1, help="number of prediction frames (unused currently)")
    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--action-emb-dim", type=int, default=4, help="dimension of action embeddings")
    parser.add_argument("--encoder-name", type=str, default="dinov2_vits14", help="DINO-V2 encoder variant (dinov2_vits14, etc)")
    parser.add_argument("--feature-key", type=str, default="x_norm_patchtokens", choices=["x_norm_patchtokens","x_norm_clstoken"], help="encoder output feature key")
    parser.add_argument("--predictor-depth", type=int, default=4, help="number of transformer layers in predictor")
    parser.add_argument("--predictor-heads", type=int, default=4, help="number of attention heads in predictor")
    parser.add_argument("--predictor-dim-head", type=int, default=32, help="dimensionality of each attention head in predictor")
    parser.add_argument("--predictor-mlp-dim", type=int, default=512, help="MLP hidden dimension in predictor")
    parser.add_argument("--predictor-dropout", type=float, default=0, help="dropout rate in predictor transformer")
    parser.add_argument("--predictor-emb-dropout", type=float, default=0, help="dropout on predictor input embeddings")
    parser.add_argument("--predictor-pool", type=str, default="attn", choices=["cls","mean","attn"], help="pooling strategy for predictor output (mean, cls, attn)")
    parser.add_argument(
        "--num-workers", type=int, default=4
    )
    parser.add_argument("--no-teacher-forcing", action="store_true", help="Disable teacher forcing (predict current frame instead of next)")
    parser.add_argument("--vicreg", action="store_true", help="Use VICReg loss")
    parser.add_argument("--vicreg-sim-coeff", type=float, default=1.0, help="VICReg similarity coefficient")
    parser.add_argument("--vicreg-var-coeff", type=float, default=1.0, help="VICReg variance coefficient")
    parser.add_argument("--vicreg-cov-coeff", type=float, default=0.01, help="VICReg covariance coefficient")
    parser.add_argument("--sched-sample-prob", type=float, default=1.0, help="Scheduled sampling: probability of using model prediction instead of ground truth for next input")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save run outputs (checkpoints, logs, plots)")
    parser.add_argument("--run-name", type=str, default=None, help="Name for wandb run")
    args = parser.parse_args()

    # transforms for DINO-V2
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.ToTensor(),
    ])

    # load raw numpy data
    states = np.load(os.path.join(args.data_dir, "states.npy"), mmap_mode='r')  # (N,T,2,H,W)
    actions = np.load(os.path.join(args.data_dir, "actions.npy"), mmap_mode='r')  # (N,T-1,2)
    print(f"Loaded states {states.shape}, actions {actions.shape}")

    # dataset & loader
    dataset = JEPATrainDataset(
        states, actions, args.num_hist, args.num_pred, transform=transform
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # model
    model = JEPAWrapper(
        encoder_name=args.encoder_name,
        feature_key=args.feature_key,
        action_emb_dim=args.action_emb_dim,
        num_hist=args.num_hist,
        pred_depth=args.predictor_depth,
        pred_heads=args.predictor_heads,
        pred_dim_head=args.predictor_dim_head,
        pred_mlp_dim=args.predictor_mlp_dim,
        pred_dropout=args.predictor_dropout,
        pred_emb_dropout=args.predictor_emb_dropout,
        pred_pool=args.predictor_pool,
    )
    device = model.device
    print(f"Using device: {device}")

    # optimizer on predictor and action encoder only
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
    )
    # learning rate scheduler with linear warmup and decay
    scheduler = LambdaLR(optimizer, lr_lambda=linear_warmup_decay(
        warmup_steps=1000,
        total_steps=len(loader) * args.epochs
    ))
    # training
    model.train()
    # create run directory
    run_dir = args.output_dir # Use the directory provided by sbatch
    os.makedirs(run_dir, exist_ok=True)
    # define run name based on argument or timestamp
    if args.run_name is not None:
        run_name = args.run_name
    else:
        run_name = f"run_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    # initialize wandb run
    wandb.init(project="jepa-final6", dir=run_dir, config=vars(args), name=run_name)
    # watch model parameters and gradients in wandb
    wandb.watch(model, log="all", log_freq=100)
    # log dataset size and total trainable parameters to wandb config
    wandb.config.train_dataset_size = len(dataset)
    wandb.config.total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # save hyperparameters
    with open(os.path.join(run_dir, "hyperparams.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # Evaluator pipeline
    pipeline = CheckpointEvalPipeline()
    pipeline.health_check()

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch_idx, (imgs, acts) in enumerate(pbar):
            B, T, C, H, W = imgs.shape
            # -----  ground-truth embeddings for t=0…T-1 -----------------------------
            with torch.no_grad():
                all_frames = F.interpolate(imgs.reshape(-1,3,H,W).to(device), 224,
                                           mode='bilinear', align_corners=False)
                all_frames = (all_frames - model.rgb_mean) / model.rgb_std
                truth_all = model._encode_frames(all_frames)        # (B*T,P,D)
            truth = truth_all.view(B, T, model.num_patches, model.embed_dim)  # (B,T,num_patches,embed_dim)

            # -----  initial frame for rollout ---------------------------------------
            init = imgs[:, :1].to(device)          # (B,1,3,H,W)
            # -----  scheduled sampling rollout -------------------------------------
            preds = model.rollout(init, acts.to(device), truth, args.sched_sample_prob)  # (B,T,num_patches,embed_dim)

            # -----  drop t=0 from loss (we never predict frame-0) --------------------
            pred = preds[:,1:]    # (B,T-1,num_patches,embed_dim)
            gold = truth[:,1:]    # (B,T-1,num_patches,embed_dim)
            if args.vicreg:
                loss = vicreg_loss(pred, gold,
                                     sim_coeff=args.vicreg_sim_coeff,
                                     var_coeff=args.vicreg_var_coeff,
                                     cov_coeff=args.vicreg_cov_coeff)
            else:
                loss = F.mse_loss(pred, gold)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # update learning rate scheduler each step
            scheduler.step()
            # log batch stats to wandb
            global_step = (epoch - 1) * len(loader) + batch_idx
            wandb.log({"batch_loss": loss.item(), "learning_rate": optimizer.param_groups[0]['lr']}, step=global_step)
            total_loss += loss.item() * B
            pbar.set_postfix({'loss': loss.item()})
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch}/{args.epochs} - loss: {avg_loss:.6f}")
        # log training loss and relevant metrics to wandb
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_loss,
            "learning_rate": optimizer.param_groups[0]['lr'],
        }, step=epoch)
        pipeline.produce_checkpoint(epoch, 
                                    lambda blob: torch.save(model.state_dict(), blob))

    # save final model
    final_save_path = os.path.join(run_dir, "final_checkpoint.pth")
    torch.save(model.state_dict(), final_save_path)
    print(f"Saved final JEPA model to {final_save_path}")
    # finish wandb run
    wandb.finish()

# Add VICReg loss function
def vicreg_loss(x, y, sim_coeff=1.0, var_coeff=1.0, cov_coeff=0.01, eps=1e-4):
    """
    Compute VICReg loss between representations x and y.
    """
    import torch
    x = x.reshape(-1, x.size(-1))
    y = y.reshape(-1, y.size(-1))
    # Invariance term (MSE)
    invariance_loss = F.mse_loss(x, y)
    # Variance term
    def variance_loss(z):
        z = z - z.mean(dim=0)
        std = torch.sqrt(z.var(dim=0) + eps)
        return torch.mean(F.relu(1 - std))
    var_loss = (variance_loss(x) + variance_loss(y)) / 2
    # Covariance term
    def covariance_loss(z):
        z = z - z.mean(dim=0)
        N, D = z.shape
        cov = (z.T @ z) / (N - 1)
        diag = torch.diag(cov)
        cov = cov - torch.diag(diag)
        return cov.pow(2).sum() / D
    cov_loss = (covariance_loss(x) + covariance_loss(y)) / 2
    return sim_coeff * invariance_loss + var_coeff * var_loss + cov_coeff * cov_loss

if __name__ == "__main__":
    main() 