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
        # action encoder
        self.action_encoder = nn.Sequential(
            nn.Linear(2, action_emb_dim),
            nn.ReLU(inplace=True),
        ).to(self.device)
        # predictor: ViT-based transformer on patch tokens
        # compute number of patches per frame (assume 224x224 input)
        num_side = 224 // self.encoder.patch_size
        num_patches = num_side * num_side
        self.predictor = ViTPredictor(
            num_patches=num_patches,
            num_frames=num_hist,
            dim=self.repr_dim,
            depth=pred_depth,
            heads=pred_heads,
            dim_head=pred_dim_head,
            mlp_dim=pred_mlp_dim,
            dropout=pred_dropout,
            emb_dropout=pred_emb_dropout,
            pool=pred_pool,
        ).to(self.device)
        self.num_hist = num_hist

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            states: (B, num_hist, C, H, W) history frames
            actions: (B, T-1, 2) actions (unused)
        Returns:
            pred_reprs: (B, num_hist, repr_dim)
        """
        states = states.to(self.device)
        B, T, C, H, W = states.shape
        # embed each frame into patch tokens
        x = states.reshape(-1, C, H, W)
        # ensure 3 channels
        if x.shape[1] == 2:
            x = torch.cat([x, x[:, 0:1]], dim=1)
        elif x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        # resize to 224×224
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        # normalize for DINO-V2
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        with torch.no_grad():
            emb = self.encoder(x)  # (B*T, num_patches, repr_dim)
        num_patches = emb.size(1)
        emb = emb.view(B, T, num_patches, self.repr_dim)
        # flatten time and patch dims
        inp = rearrange(emb, 'b t p d -> b (t p) d')
        # apply ViT predictor; fallback to mean-pooled repr on OOM
        try:
            out = self.predictor(inp)  # (B, T*num_patches, repr_dim)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                torch.cuda.empty_cache()
                # fallback: collapse spatial tokens directly
                reprs = emb.mean(dim=2)  # (B, T, repr_dim)
                return reprs
            else:
                raise
        out = out.view(B, T, num_patches, self.repr_dim)
        # mean-pool spatial tokens per frame
        reprs = out.mean(dim=2)  # (B, T, repr_dim)
        return reprs

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
        frames = self.states[traj_idx, start:end]  # (W, C, H, W)
        acts = self.actions[traj_idx, start:end - 1]  # (W-1, 2)
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
    parser.add_argument("--batch-size", type=int, default=16, help="training batch size")
    parser.add_argument("--num-hist", type=int, default=16, help="history frames for JEPA prediction")
    parser.add_argument("--num-pred", type=int, default=1, help="number of prediction frames (unused currently)")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--action-emb-dim", type=int, default=4, help="dimension of action embeddings")
    parser.add_argument("--encoder-name", type=str, default="dinov2_vits14", help="DINO-V2 encoder variant (dinov2_vits14, etc)")
    parser.add_argument("--feature-key", type=str, default="x_norm_patchtokens", choices=["x_norm_patchtokens","x_norm_clstoken"], help="encoder output feature key")
    parser.add_argument("--predictor-depth", type=int, default=2, help="number of transformer layers in predictor")
    parser.add_argument("--predictor-heads", type=int, default=4, help="number of attention heads in predictor")
    parser.add_argument("--predictor-dim-head", type=int, default=32, help="dimensionality of each attention head in predictor")
    parser.add_argument("--predictor-mlp-dim", type=int, default=512, help="MLP hidden dimension in predictor")
    parser.add_argument("--predictor-dropout", type=float, default=0.0, help="dropout rate in predictor transformer")
    parser.add_argument("--predictor-emb-dropout", type=float, default=0.0, help="dropout on predictor input embeddings")
    parser.add_argument("--predictor-pool", type=str, default="mean", choices=["cls","mean"], help="pooling strategy for predictor output")
    parser.add_argument(
        "--save-path", type=str, default="jepa_model_small_16hist.pth"
    )
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    # transforms for DINO-V2
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
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
        list(model.action_encoder.parameters())
        + list(model.predictor.parameters()),
        lr=args.lr,
    )

    # training
    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        # iterate with progress bar per epoch
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
        for imgs, acts in pbar:
            # imgs: (B,W,3,H,W); acts: (B, W-1,2)
            B, W, C, H, Wp = imgs.shape
            # use history frames for ViTPredictor
            hist = imgs[:, :args.num_hist].to(device)  # (B, num_hist, 3, H, W)
            acts = acts.to(device)
            preds = model(hist, acts)  # (B, num_hist, repr_dim)
            # compute target embeddings for history frames [1:]
            frames = hist[:, 1:].reshape(-1, C, H, Wp).to(device)
            if frames.shape[1] == 2:
                frames = torch.cat([frames, frames[:, 0:1]], dim=1)
            elif frames.shape[1] == 1:
                frames = frames.repeat(1, 3, 1, 1)
            frames = F.interpolate(frames, size=(224, 224), mode='bilinear', align_corners=False)
            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
            frames = (frames - mean) / std
            with torch.no_grad():
                emb = model.encoder(frames)  # (B*(num_hist-1), num_patches, repr_dim)
                targ = emb.mean(dim=1)  # (B*(num_hist-1), repr_dim)
            targ = targ.view(B, args.num_hist - 1, model.repr_dim)
            pr = preds[:, 1:]  # (B, num_hist-1, repr_dim)
            # compute MSE on history steps
            loss = F.mse_loss(pr, targ)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * B
            # show current batch loss
            pbar.set_postfix({'loss': loss.item()})
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch}/{args.epochs} - loss: {avg_loss:.6f}")

    # save final model
    torch.save(model.state_dict(), args.save_path)
    print(f"Saved JEPA model to {args.save_path}")

if __name__ == "__main__":
    main() 