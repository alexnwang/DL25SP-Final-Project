import torch
from torch.utils.data import DataLoader
from models import JEPA_SplitEncoder_VICReg, JEPA_SplitEncoder_VICRegV2, JEPA_SplitEncoder_CombinedLoss, JEPA_SplitEncoder_CombinedLossNoPool
from configs import ConfigSplitJEPA
from dataset import WallDataset
from evaluator import ProbingEvaluator
from tqdm import tqdm
import os
from torch.utils.data import random_split

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

def create_dataloaders(config):
    full_dataset = WallDataset(config.data_path, probing=False)

    train_ratio = 0.9 # 0.98 is also a good option
    train_size = int(train_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )


    return train_loader, val_loader

def train_one_epoch(model, dataloader, device):
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc="Training")

    for batch in pbar:
        loss = model.training_step(batch, device)
        total_loss += loss

    avg_loss = total_loss / len(dataloader)
    return avg_loss

@torch.no_grad()
def validate_one_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc="Validation")

    for batch in pbar:
        loss = model.validation_step(batch)
        total_loss += loss

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def save_checkpoint(model, epoch, train_loss, val_loss, save_dir="weights"):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model.name}_epoch{epoch}_trainloss_{train_loss}_valloss_{val_loss}.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Saved checkpoint: {save_path}")

def main():
    config = ConfigSplitJEPA()
    device = get_device()

    model = JEPA_SplitEncoder_CombinedLossNoPool(config).to(device)
    print(f"Model: {model.name} | Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    train_loader, val_loader = create_dataloaders(config)
    config.steps_per_epoch = len(train_loader)

    best_val_loss = float("inf")

    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch+1}/{config.epochs}")

        train_loss = train_one_epoch(model, train_loader, device)
        val_loss = validate_one_epoch(model, val_loader, device)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
        save_checkpoint(model, epoch, train_loss, val_loss)

if __name__ == "__main__":
    main()
