# utils/optim.py
import math, torch
from torch.optim      import Adam, AdamW, SGD
from torch.optim.lr_scheduler import (
    LambdaLR, CosineAnnealingLR, OneCycleLR, StepLR
)


# --------------------------------------------------------------------- #
#  Helper: total number of training steps
# --------------------------------------------------------------------- #
def _total_steps(cfg):
    if hasattr(cfg, "total_steps"):
        return cfg.total_steps
    assert hasattr(cfg, "epochs") and hasattr(cfg, "steps_per_epoch"), (
        "Need either cfg.total_steps or (epochs & steps_per_epoch)!"
    )
    return cfg.epochs * cfg.steps_per_epoch


# --------------------------------------------------------------------- #
#  get_optimizer
# --------------------------------------------------------------------- #
def get_optimizer(cfg, params):
    """
    Parameters
    ----------
    cfg    : JEPAConfig  (must have lr, weight_decay, optimizer fields)
    params : iterable[torch.nn.Parameter]

    Returns
    -------
    torch.optim.Optimizer
    """
    opt_name      = getattr(cfg, "optimizer", "adamw").lower()
    lr            = getattr(cfg, "learning_rate", 1e-4)
    wd            = getattr(cfg, "weight_decay", 1e-2)
    betas         = getattr(cfg, "betas", (0.9, 0.999))
    momentum      = getattr(cfg, "momentum", 0.9)

    if opt_name in ("adam", "adamw"):
        OptCls = AdamW if opt_name == "adamw" else Adam
        return OptCls(params, lr=lr, betas=betas, weight_decay=wd)

    if opt_name == "sgd":
        return SGD(params, lr=lr, momentum=momentum, weight_decay=wd)

    raise ValueError(f"Unknown optimizer: {opt_name}")


# --------------------------------------------------------------------- #
#  get_scheduler
# --------------------------------------------------------------------- #
def get_scheduler(optim, cfg):
    """
    Parameters
    ----------
    optim : torch.optim.Optimizer
    cfg   : JEPAConfig  (needs scheduler, epochs, steps_per_epoch, etc.)

    Returns
    -------
    torch.optim.lr_scheduler._LRScheduler
    """
    sched_name = getattr(cfg, "scheduler", "cosine").lower()
    warmup_pct = getattr(cfg, "warmup_pct", 0.05)     # 5 % by default
    total      = _total_steps(cfg)
    warmup     = int(total * warmup_pct)

    # ---- 1. cosine with linear warm‑up -------------------------------
    if sched_name == "cosine":
        def lr_lambda(step):
            if step < warmup:                              # warm‑up
                return step / float(max(1, warmup))
            progress = (step - warmup) / float(max(1, total - warmup))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return LambdaLR(optim, lr_lambda=lr_lambda)

    # ---- 2. One‑cycle -------------------------------------------------
    if sched_name == "onecycle":
        return OneCycleLR(
            optim,
            max_lr=getattr(cfg, "learning_rate", 1e-4),
            total_steps=total,
            pct_start=warmup / total,
            anneal_strategy="cos")

    # ---- 3. Step decay -----------------------------------------------
    if sched_name == "step":
        step_size = getattr(cfg, "step_size", total // 3)
        gamma     = getattr(cfg, "gamma", 0.2)
        return StepLR(optim, step_size=step_size, gamma=gamma)

    raise ValueError(f"Unknown scheduler: {sched_name}")
