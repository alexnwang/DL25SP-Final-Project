from typing import NamedTuple, List, Any, Optional, Dict
from itertools import chain
from dataclasses import dataclass
import itertools
import os
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import numpy as np
from matplotlib import pyplot as plt

from schedulers import Scheduler, LRSchedule
from models import Prober, build_mlp
from configs import ConfigBase

from dataset import WallDataset
from normalizer import Normalizer


@dataclass
class ProbingConfig(ConfigBase):
    probe_targets: str = "locations"
    lr: float = 0.0002
    epochs: int = 20
    schedule: LRSchedule = LRSchedule.Cosine
    sample_timesteps: int = 30
    prober_arch: str = "256"


class ProbeResult(NamedTuple):
    model: torch.nn.Module
    average_eval_loss: float
    eval_losses_per_step: List[float]
    plots: List[Any]


default_config = ProbingConfig()


def location_losses(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    assert pred.shape == target.shape
    mse = (pred - target).pow(2).mean(dim=0)
    return mse


class ProbingEvaluator:
    def __init__(
        self,
        device: "cuda",
        model: torch.nn.Module,
        probe_train_ds,
        probe_val_ds: dict,
        config: ProbingConfig = default_config,
        quick_debug: bool = False,
    ):
        self.device = device
        self.config = config

        self.model = model
        self.model.eval()

        self.quick_debug = quick_debug

        self.ds = probe_train_ds
        self.val_ds = probe_val_ds

        self.normalizer = Normalizer()

    def train_pred_prober(self):
        """
        Probes whether the predicted embeddings capture the future locations
        """
        repr_dim = self.model.embed_dim
        dataset = self.ds
        model = self.model

        config = self.config
        epochs = config.epochs

        if self.quick_debug:
            epochs = 1
        test_batch = next(iter(dataset))

        prober_output_shape = getattr(test_batch, "locations")[0, 0].shape
        prober = Prober(
            repr_dim,
            config.prober_arch,
            output_shape=prober_output_shape,
        ).to(self.device)

        all_parameters = []
        all_parameters += list(prober.parameters())

        optimizer_pred_prober = torch.optim.Adam(all_parameters, config.lr)

        step = 0

        batch_size = dataset.batch_size
        batch_steps = None

        scheduler = Scheduler(
            schedule=self.config.schedule,
            base_lr=config.lr,
            data_loader=dataset,
            epochs=epochs,
            optimizer=optimizer_pred_prober,
            batch_steps=batch_steps,
            batch_size=batch_size,
        )

        for epoch in tqdm(range(epochs), desc=f"Probe prediction epochs"):
            for batch in tqdm(dataset, desc="Probe prediction step"):
                ################################################################################
                init_state = batch.states[:, :1].to(self.device)
                actions    = batch.actions.to(self.device)
                with torch.no_grad():
                    emb_seq = model.rollout(init_state, actions)   # (B, T, P, D)
                pred_encs = emb_seq.mean(dim=2)[:, 1:]             # (B, T-1, D)
                pred_encs = pred_encs.transpose(0, 1).contiguous() # (T-1, B, D)

                # On first step, compute & stash training μ/σ
                if step == 0:
                    μ = pred_encs.mean((0, 1), keepdim=True)
                    σ = pred_encs.std((0, 1), keepdim=True).clamp_min(1e-6)
                    self.feat_mean = μ.detach()
                    self.feat_std  = σ.detach()

                # Apply z-score with stored stats
                pred_encs = (pred_encs - self.feat_mean) / self.feat_std
                ################################################################################

                pred_encs = pred_encs.detach()

                n_steps = pred_encs.shape[0]
                bs = pred_encs.shape[1]

                losses_list = []

                target = batch.locations.cuda()[:,1:]
                target = self.normalizer.normalize_location(target)

                if (
                    config.sample_timesteps is not None
                    and config.sample_timesteps < n_steps
                ):
                    sample_shape = (config.sample_timesteps,) + pred_encs.shape[1:]
                    # we only randomly sample n timesteps to train prober.
                    # we most likely do this to avoid OOM
                    sampled_pred_encs = torch.empty(
                        sample_shape,
                        dtype=pred_encs.dtype,
                        device=pred_encs.device,
                    )

                    sampled_target_locs = torch.empty(bs, config.sample_timesteps, 2)

                    for i in range(bs):
                        indices = torch.randperm(n_steps)[: config.sample_timesteps]
                        sampled_pred_encs[:, i, :] = pred_encs[indices, i, :]
                        sampled_target_locs[i, :] = target[i, indices]

                    pred_encs = sampled_pred_encs
                    target = sampled_target_locs.cuda()

                pred_locs = torch.stack([prober(x) for x in pred_encs], dim=1)
                losses = location_losses(pred_locs, target)
                per_probe_loss = losses.mean()

                if step % 100 == 0:
                    print(f"normalized pred locations loss {per_probe_loss.item()}")

                losses_list.append(per_probe_loss)
                optimizer_pred_prober.zero_grad()
                loss = sum(losses_list)
                loss.backward()
                optimizer_pred_prober.step()

                lr = scheduler.adjust_learning_rate(step)

                step += 1

                if self.quick_debug and step > 2:
                    break

        return prober

    @torch.no_grad()
    def evaluate_all(
        self,
        prober,
    ):
        """
        Evaluates on all the different validation datasets
        """
        avg_losses = {}

        for prefix, val_ds in self.val_ds.items():
            avg_losses[prefix] = self.evaluate_pred_prober(
                prober=prober,
                val_ds=val_ds,
                prefix=prefix,
            )

        return avg_losses

    @torch.no_grad()
    def evaluate_pred_prober(
        self,
        prober,
        val_ds,
        prefix="",
    ):
        quick_debug = self.quick_debug
        config = self.config

        model = self.model
        probing_losses = []
        prober.eval()

        for idx, batch in enumerate(tqdm(val_ds, desc="Eval probe pred")):
            ################################################################################
            init_state = batch.states[:, :1].to(self.device)
            actions    = batch.actions.to(self.device)
            emb_seq = model.rollout(init_state, actions)       # (B, T, P, D)
            pred_encs = emb_seq.mean(dim=2)[:, 1:]               # (B, T-1, D)
            pred_encs = pred_encs.transpose(0, 1).contiguous()   # (T-1, B, D)

            # Apply exactly the same normalization from training
            pred_encs = (pred_encs - self.feat_mean) / self.feat_std
            ################################################################################

            target = batch.locations.cuda()[:,1:]
            target = self.normalizer.normalize_location(target)

            pred_locs = torch.stack([prober(x) for x in pred_encs], dim=1)
            losses = location_losses(pred_locs, target)
            probing_losses.append(losses.cpu())

        losses_t = torch.stack(probing_losses, dim=0).mean(dim=0)
        losses_t = self.normalizer.unnormalize_mse(losses_t)

        losses_t = losses_t.mean(dim=-1)
        average_eval_loss = losses_t.mean().item()

        return average_eval_loss