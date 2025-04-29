from dataset import create_wall_dataloader
from evaluator import ProbingEvaluator
import torch
import glob
import argparse
from pipeline import CheckpointEvalPipeline
from train_jepa import JEPAWrapper
import wandb


def get_device():
    """Check for GPU availability."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device


def load_data(device):
    data_path = "/gpfs/scratch/wz1492/data"

    probe_train_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/train",
        probing=True,
        device=device,
        train=True,
    )

    probe_val_normal_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/val",
        probing=True,
        device=device,
        train=False,
    )

    probe_val_wall_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_wall/val",
        probing=True,
        device=device,
        train=False,
    )

    probe_val_ds = {
        "normal": probe_val_normal_ds,
        "wall": probe_val_wall_ds,
    }

    return probe_train_ds, probe_val_ds


def load_model(device, args):
    """Load JEPAWrapper from checkpoint and instantiate with provided args."""
    checkpoint = torch.load(args.checkpoint, map_location=device)
    # infer action_emb_dim from first linear layer of action_encoder MLP
    action_emb_dim = checkpoint['action_encoder.0.weight'].shape[0]
    # instantiate JEPAWrapper with training hyperparameters
    model = JEPAWrapper(
        encoder_name=args.encoder_name,
        feature_key=args.feature_key,
        action_emb_dim=action_emb_dim,
        num_hist=args.num_hist,
        pred_depth=args.predictor_depth,
        pred_heads=args.predictor_heads,
        pred_dim_head=args.predictor_dim_head,
        pred_mlp_dim=args.predictor_mlp_dim,
        pred_dropout=args.predictor_dropout,
        pred_emb_dropout=args.predictor_emb_dropout,
        pred_pool=args.predictor_pool,
    )
    # load weights non-strictly to skip old predictor parameter mismatches
    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    if missing or unexpected:
        print(f"Warning: missing keys in state_dict: {missing}")
        print(f"Warning: unexpected keys in state_dict: {unexpected}")
    model.to(device)
    model.eval()
    return model


def evaluate_model(device, model, probe_train_ds, probe_val_ds):
    evaluator = ProbingEvaluator(
        device=device,
        model=model,
        probe_train_ds=probe_train_ds,
        probe_val_ds=probe_val_ds,
        quick_debug=False,
    )

    prober = evaluator.train_pred_prober()

    avg_losses = evaluator.evaluate_all(prober=prober)

    logs = {}
    for probe_attr, loss in avg_losses.items():
        logs[f"{probe_attr} loss"]= loss
    wandb.logs(logs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate JEPA model")
    # model checkpoint and hyperparameters
    parser.add_argument("--checkpoint", type=str, default="/gpfs/scratch/wz1492/DL25SP-Final-Project/runs/63423646/task5_size-tiny_d4_h4_hd32_mlp512_do0.05_sched0.5_lr5.67774e-05_seed1005/model_epoch_7.pth", help="path to JEPA model checkpoint")
    parser.add_argument("--encoder-name", type=str, default="dinov2_vits14", help="DINO-V2 encoder variant used in training")
    parser.add_argument("--feature-key", type=str, default="x_norm_patchtokens", choices=["x_norm_patchtokens","x_norm_clstoken"], help="encoder feature key used in training")
    parser.add_argument("--num-hist", type=int, default=16, help="history length used in training")
    parser.add_argument("--predictor-depth", type=int, default=4, help="number of transformer layers in predictor")
    parser.add_argument("--predictor-heads", type=int, default=4, help="number of attention heads in predictor")
    parser.add_argument("--predictor-dim-head", type=int, default=32, help="dimensionality of each attention head in predictor")
    parser.add_argument("--predictor-mlp-dim", type=int, default=512, help="MLP hidden dimension in predictor")
    parser.add_argument("--predictor-dropout", type=float, default=0.0, help="dropout rate in predictor transformer")
    parser.add_argument("--predictor-emb-dropout", type=float, default=0.0, help="dropout on predictor input embeddings")
    parser.add_argument("--predictor-pool", type=str, default="attn", choices=["cls","mean","attn"], help="predictor pool used in training (mean, cls, attn)")
    args = parser.parse_args()
    device = get_device() 

    pipeline = CheckpointEvalPipeline()
    pipeline.health_check()

    pipeline.subscribe_checkpoint(lambda checkpoint_path: args.update({"checkpoint": checkpoint_path}))
    model = load_model(device, args)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")

    probe_train_ds, probe_val_ds = load_data(device)
    evaluate_model(device, model, probe_train_ds, probe_val_ds)
