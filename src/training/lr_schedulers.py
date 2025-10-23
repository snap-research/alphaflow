import torch
from torch.optim.lr_scheduler import LRScheduler

from src.structs import EasyDict

#----------------------------------------------------------------------------

def create_composite_lr_scheduler(optimizer: torch.optim.Optimizer, cfg: EasyDict) -> LRScheduler:
    num_cosine_updates = cfg.max_steps - cfg.num_warmup_steps
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0 / cfg.num_warmup_steps, end_factor=1.0, total_iters=cfg.num_warmup_steps)
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_cosine_updates, eta_min=cfg.final_lr)
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup_scheduler, main_scheduler], milestones=[cfg.num_warmup_steps])

    return lr_scheduler

#----------------------------------------------------------------------------
