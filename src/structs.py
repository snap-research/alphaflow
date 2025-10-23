"""
This file contains the definition of some useful structs
to aggregate variables into convenient containers to move around.

Note:
1. They are intentionally defined in a standalone file to avoid circular imports.
2. They are intentionally defined as namedtuples (where possible) to make them immutable, simple and clear. Ideally, they shouldn't have any methods (static ones are fine).

TODO: maybe we should define these structs in the places where they are used?
TODO: maybe we should add validations to the properties?
TODO: maybe we should use dataclasses instead of namedtuples?
"""
import os
import copy
import operator
from enum import Enum
from dataclasses import dataclass, asdict
from typing import Any, Union, Optional, Callable
from collections import namedtuple

import numpy as np
import torch
from torch.distributed.fsdp import ShardingStrategy
from beartype import beartype
from omegaconf import DictConfig, ListConfig

import src.utils.distributed as dist
from src.utils.os_utils import find_snapshots

#------------------------------------------------------------------------------------------
# A simple dict class with attribute-style access.

class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]

    @staticmethod
    def init_recursively(value: Union[dict, DictConfig, "EasyDict"], init_collections: bool = True, target_class=None) -> "EasyDict":
        # TODO: using `target_class` for just a single use case looks dirty as hell.
        target_class = EasyDict if target_class is None else target_class
        if isinstance(value, (dict, DictConfig, EasyDict)):
            # We pack a new dict via the for loop to be able to use non-string property names
            out = target_class()
            for k, v in value.items():
                out[k] = EasyDict.init_recursively(v, target_class=target_class)
            return out
        elif isinstance(value, (list, set, tuple, ListConfig)) and init_collections:
            # conversions = {list: list, set: set, tuple: tuple, ListConfig: list}
            # return conversions[type(value)]([EasyDict.init_recursively(v) for v in value])
            conversions = {list: list, set: set, tuple: tuple, ListConfig: list}
            return [EasyDict.init_recursively(v, target_class=target_class) for v in value]
        else:
            return value

    def to_dict(self) -> dict:
        return self.init_recursively(self, target_class=dict)

    def copy(self) -> "EasyDict":
        return copy.deepcopy(self)

#----------------------------------------------------------------------------
# A TensorDict-like class, but which only supports torch tensors and forces them to have the same batch dimension.

class TensorGroup(EasyDict):
    """
    Sometimes, it is very convenient to group tensors into a group (i.e. video + audio).
    You can slice/split TensorGroup in the same manner as you would do with normal torch tensors
    The tensors are aligned via the first axis.

    Caution: when updating the properties, it's your responsibility to
    make sure that the shapes remain to be correct (i.e. of the same length).
    """
    def validate(self):
        keys = list(self.keys())
        values = list(self.values())

        assert len(keys) == len(values)
        assert all(isinstance(key, str) for key in keys), f"Keys can only be strings (otherwise, we confuse them with idx): {keys}"
        assert all((isinstance(t, torch.Tensor) or isinstance(t, TensorGroup)) for t in values), f"Wrong types for values: {dict(zip(keys, [type(v) for v in values]))}"
        assert all(len(t) == len(values[0]) for t in values), f"Wrong shapes: {dict(zip(keys, [v.shape for v in values]))}"
        assert all(t.device == values[0].device for t in values), f"All tensor should be on the same device, but got: {dict(zip(keys, [v.device for v in values]))}"

    def __getitem__(self, item: Any):
        if isinstance(item, str):
            return super(TensorGroup, self).__getitem__(item)
        else:
            return TensorGroup(**{k: v[item] for k, v in self.items()})

    def __len__(self) -> int:
        return len(list(self.values())[0])

    def split(self, group_size: int) -> list["TensorGroup"]:
        result = []
        for group_idx in range((len(self) + group_size - 1) // group_size):
            result.append(self[group_idx * group_size: (group_idx + 1) * group_size])
        return result

    def reduce_max(self) -> torch.Tensor:
        """
        This method is useless on its own and we use it simply to aggregate some values
        from all the tensors to have DDP consistency
        """
        return torch.stack([v.max() for k, v in self.items()]).max() # [1]

    def reduce_mean(self) -> torch.Tensor:
        """
        This method is useless on its own and we use it simply to aggregate some values
        from all the tensors to have DDP consistency
        """
        return self.sum() / self.numel() # [1]

    def reduce_sum(self) -> torch.Tensor:
        return torch.stack([v.sum() for k, v in self.items()]).sum() # [1]

    def numel(self) -> int:
        return sum([v.numel() for k, v in self.items()])

    @property
    def device(self):
        return next(iter(self.items()))[1].device

    @property
    def shape(self) -> EasyDict[str, torch.Size]:
        return EasyDict({k: v.shape for k, v in self.items()})

    def reshape_each(self, reshaper: Callable) -> "TensorGroup":
        return TensorGroup(**{k: v.reshape(reshaper(v)) for k, v in self.items()})

    def view_each(self, reshaper: Callable) -> "TensorGroup":
        return TensorGroup(**{k: v.view(reshaper(v)) for k, v in self.items()})

    def view(self, *args, **kwargs) -> "TensorGroup":
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            return self.view_each(lambda x: x.view(*args[0], **kwargs))
        elif isinstance(args[0], (EasyDict, dict)):
            shapes = args[0]
            assert set(shapes.keys()) == set(self.keys()), f"Keys should be the same: {shapes.keys()} vs {self.keys()}"
            return TensorGroup(**{k: v.view(*shapes[k], **kwargs) for k, v in self.items()})

    @staticmethod
    def cat(tgroups: list["TensorGroup"], dim: int=0):
        keys_set = set(tgroups[0].keys())
        assert [set(tg.keys()) == keys_set for tg in tgroups], f"Keys should be the same: {[list(tg.keys()) for tg in tgroups]}"
        return TensorGroup(**{k: torch.cat([tg[k] for tg in tgroups], dim=dim) for k in keys_set})

    @staticmethod
    def stack(tgroups: list["TensorGroup"], dim: int=0):
        keys_set = set(tgroups[0].keys())
        assert [set(tg.keys()) == keys_set for tg in tgroups], f"Keys should be the same: {[list(tg.keys()) for tg in tgroups]}"
        return TensorGroup(**{k: torch.stack([tg[k] for tg in tgroups], dim=dim) for k in keys_set})

    @staticmethod
    def map(tg: "TensorGroup", fn: Callable, *args, **kwargs) -> "TensorGroup":
        return TensorGroup(**{k: fn(v, *args, **kwargs) for k, v in tg.items()})

def _unary_factory(name):
    def fn(self, *args, **kwargs):
        return TensorGroup(**{k: getattr(v, name)(*args, **kwargs) for k, v in self.items()})
    fn.__name__ = name
    fn.__doc__  = f"Apply torch.Tensor.{name} to each value."
    return fn

UNARY_METHODS = [
    "reshape", "flatten", "squeeze", "unsqueeze", "sum", "max",
    "permute", "repeat", "repeat_interleave", "clamp", "float", "sigmoid",
    "detach", "cpu", "to", "contiguous", "sqrt", "pow", "mean", "fill_", "exp",
]

for _m in UNARY_METHODS:
    setattr(TensorGroup, _m, _unary_factory(_m))

def _binary_factory(op, reverse: bool=False):
    def fn(self, other):
        if isinstance(other, TensorGroup):
            keys = self.keys()
            assert set(keys) == set(other.keys())
            return TensorGroup(**{k: op(other[k], self[k]) if reverse else op(self[k], other[k]) for k in keys})
        else:
            return TensorGroup(**{k: op(other, v) if reverse else op(v, other) for k, v in self.items()})
    return fn

TensorGroup.__add__      = _binary_factory(operator.add)
TensorGroup.__radd__     = TensorGroup.__add__
TensorGroup.__sub__      = _binary_factory(operator.sub)
TensorGroup.__rsub__     = _binary_factory(operator.sub, reverse=True)
TensorGroup.__sub__      = _binary_factory(operator.sub)
TensorGroup.__mul__      = _binary_factory(operator.mul)
TensorGroup.__rmul__     = TensorGroup.__mul__
TensorGroup.__pow__      = _binary_factory(operator.pow)
TensorGroup.__truediv__  = _binary_factory(operator.truediv)   # tg / x
TensorGroup.__floordiv__ = _binary_factory(operator.floordiv)
TensorGroup.__rtruediv__ = _binary_factory(operator.truediv, reverse=True)  # x / tg
TensorGroup.__rfloordiv__ = _binary_factory(operator.floordiv, reverse=True)

#----------------------------------------------------------------------------

Nets = namedtuple('Nets', ['net', 'ema'])
Optimizers = namedtuple('Optimizers', ['optimizer', 'lr_scheduler', 'grad_scaler'])

#----------------------------------------------------------------------------

# A container to carry out simple statistics for the latents.
BasicStats = namedtuple('BasicStats', ['mean', 'std', 'min', 'max', 'num_samples_used'])

@dataclass(frozen=True)
class BasicStatsAgg:
    mean_min: float
    mean_mean: float
    mean_max: float
    mean_abs_min: float
    mean_abs_mean: float
    mean_abs_max: float
    std_mean: float
    std_min: float
    std_max: float
    min_min: float
    max_max: float
    num_samples_used: int

    @staticmethod
    def infer_from_basic_stats(stats: BasicStats) -> "BasicStatsAgg":
        return BasicStatsAgg(
            mean_min=stats.mean.min().item(),
            mean_mean=stats.mean.mean().item(),
            mean_max=stats.mean.max().item(),
            mean_abs_min=np.abs(stats.mean).min().item(),
            mean_abs_mean=np.abs(stats.mean).mean().item(),
            mean_abs_max=np.abs(stats.mean).max().item(),
            std_mean=stats.std.mean().item(),
            std_min=stats.std.min().item(),
            std_max=stats.std.max().item(),
            min_min=stats.min.min().item(),
            max_max=stats.max.max().item(),
            num_samples_used=stats.num_samples_used,
        )

    def __str__(self) -> str:
        return ', '.join([(f'{k}: {v:.4f}' if isinstance(v, float) else f'{k}: {v}') for k, v in asdict(self).items()])

def aggregate_stats_across_ranks(stats: BasicStats) -> tuple[torch.Tensor, torch.Tensor]:
    means = dist.gather_concat(stats.mean.unsqueeze(0)) # [num_ranks, t, c, h, w]
    stds = dist.gather_concat(stats.std.unsqueeze(0)) # [num_ranks, t, c, h, w]
    mean_global = torch.mean(means, dim=0) # [t, c, h, w]
    std_global = aggregate_variance(means, stds.pow(2), stats.num_samples_used).sqrt() # [t, c, h, w]
    min_global = dist.gather_concat(stats.min.unsqueeze(0)).min(dim=0).values
    max_global = dist.gather_concat(stats.max.unsqueeze(0)).max(dim=0).values
    stats_global = BasicStats(mean=mean_global, std=std_global, min=min_global, max=max_global, num_samples_used=stats.num_samples_used * dist.get_world_size())
    return stats_global

def aggregate_variance(subset_means: torch.Tensor, subset_vars: torch.Tensor, subset_size: int) -> torch.Tensor:
    """
    Computes the variance of a set using the means/variances of its subsets. Assumes that each subset has the same size.
    Params: subset_means [num_subsets, d_1, ..., d_n], subset_vars [num_subsets, d_1, ..., d_n], subset_size [1]
    """
    assert subset_size > 1, f"subset_size={subset_size} must be greater than 1 since otherwise it was impossible to calculate the variance."
    num_subsets = len(subset_means)
    dtype = subset_vars.dtype
    subset_means, subset_vars = subset_means.double(), subset_vars.double() # [num_subsets, d_1, ..., d_n], [num_subsets, d_1, ..., d_n]
    subset_varraws = var2rawvar(subset_vars, subset_means, subset_size)
    var_raw_global = subset_varraws.sum(dim=0) # [d_1, ..., d_n]
    mean_global = subset_means.mean(dim=0) # [d_1, ..., d_n]
    var_global = rawvar2var(var_raw_global, mean_global, subset_size * num_subsets)
    return var_global.to(dtype)

def var2rawvar(var, mean, num_items: int):
    return (var + mean.pow(2)) * num_items

def rawvar2var(rawvar, mean, num_items: int):
    return (rawvar / num_items) - mean.pow(2)

#----------------------------------------------------------------------------
# Token types for videos.

TokenTypes = ['QUERY', 'MISSING', 'CONTEXT']
TokenType = Enum('TokenType', TokenTypes)
TensorLike = torch.Tensor | TensorGroup
BatchInfo = namedtuple('BatchInfo', ['batch_size', 'batch_gpu', 'num_accum_rounds'])

#----------------------------------------------------------------------------

@dataclass(frozen=True)
class ResumeConfig:
    ckpt_path: Optional[str] = None
    whole_state: bool = True
    allow_missing_keys: bool = False
    allow_unexpected_keys: bool = False
    allow_missing_ema: bool = False
    allow_missing_optim: bool = False
    allow_missing_stats: bool = False
    allow_shape_mismatch: bool = False
    allow_missing_keys_regex: Optional[str] = None
    keys_aug: Optional[list[str, str]] = None

    def is_strict_loading(self) -> bool:
        return not (self.allow_unexpected_keys or self.allow_missing_keys or self.allow_shape_mismatch or self.allow_missing_keys_regex is not None)

    @staticmethod
    def from_scratch() -> "ResumeConfig":
        return ResumeConfig(ckpt_path=None, whole_state=False)

    @staticmethod
    def from_ckpt(ckpt_path: str, resume_cfg: EasyDict, allow_missing_extra_state: bool=False) -> "ResumeConfig":
        return ResumeConfig(
            ckpt_path=ckpt_path,
            whole_state=resume_cfg.whole_state,
            allow_missing_keys=resume_cfg.allow_missing_keys,
            allow_unexpected_keys=resume_cfg.allow_unexpected_keys,
            allow_missing_ema=resume_cfg.allow_missing_ema or allow_missing_extra_state,
            allow_missing_optim=resume_cfg.allow_missing_optim or allow_missing_extra_state,
            allow_missing_stats=allow_missing_extra_state,
            allow_shape_mismatch=False, # Let's not allow shape mismatches for now.
        )

    @beartype
    @staticmethod
    def init_from_experiment_cfg(exp_cfg: EasyDict) -> "ResumeConfig":
        # Initialize the snapshot to resume from (if possible).
        snapshots = find_snapshots(exp_cfg.output_dir)
        if len(snapshots) > 0:
            ckpt_path = os.path.join(exp_cfg.output_dir, snapshots[-1])
            dist.loginfo0(f'Found {len(snapshots)} snapshots in {exp_cfg.output_dir}. Will resume training from the latest one: {ckpt_path}.')
            resume_cfg = ResumeConfig.from_ckpt(ckpt_path, exp_cfg.training.resume, allow_missing_extra_state=False)
        elif exp_cfg.training.resume.on_start_ckpt_path is not None:
            dist.loginfo0(f'Didnt find any snapshots in {exp_cfg.output_dir}. Will resume training from the specified checkpoint: {exp_cfg.training.resume.on_start_ckpt_path}.')
            if exp_cfg.model.ema_rampup_ratio is not None:
                dist.logwarn0(f'When fine-tuning, it is better to disable ema_rampup_ratio, but got ema_rampup_ratio={exp_cfg.model.ema_rampup_ratio}')
            resume_cfg = ResumeConfig.from_ckpt(exp_cfg.training.resume.on_start_ckpt_path, exp_cfg.training.resume, allow_missing_extra_state=exp_cfg.training.resume.allow_missing_extra_state_on_start)
        else:
            dist.loginfo0(f'No snapshots found in {exp_cfg.output_dir}. Training from scratch.')
            resume_cfg = ResumeConfig.from_scratch()

        return resume_cfg

#----------------------------------------------------------------------------
# Some constants.

DTYPE_STRING_TO_DTYPE = {
    'float32': torch.float32,
    'float64': torch.float64,
    'float16': torch.float16,
    'int32': torch.int32,
    'int64': torch.int64,
    'bfloat16': torch.bfloat16,
    None: None,
}

FSDP_STRATEGY_STRING_TO_FSDP_STRATEGY = {
    'full_shard': ShardingStrategy.FULL_SHARD,
    'shard_grad_op': ShardingStrategy.SHARD_GRAD_OP,
    'no_shard': ShardingStrategy.NO_SHARD,
    'hybrid_shard': ShardingStrategy.HYBRID_SHARD,
}

NUMPY_INTEGER_TYPES = [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]
NUMPY_FLOAT_TYPES = [np.float16, np.float32, np.float64, np.single, np.double]

#----------------------------------------------------------------------------

class DataSampleType(Enum):
    IMAGE = 'IMAGE'
    VIDEO = 'VIDEO'
    IMAGE_LATENT = 'IMAGE_LATENT'
    VIDEO_LATENT = 'VIDEO_LATENT'

    def __str__(self) -> str:
        return self.value

    @staticmethod
    def from_str(type_str: str) -> "DataSampleType":
        return DataSampleType[type_str.upper()]

#----------------------------------------------------------------------------
# Loss-related structs.

class BaseLoss(torch.nn.Module):
    pass

class LossPhase(Enum):
    Gen = 'G'
    GenLipReg = 'Gr1'
    GenAll = 'Gall'
    GenAndDiscr = 'GD'
    Discr = 'D'

    def has_gen_update(self) -> bool:
        return self in (LossPhase.Gen, LossPhase.GenLipReg, LossPhase.GenAll, LossPhase.GenAndDiscr)

    def has_discr_update(self) -> bool:
        return self in (LossPhase.Discr, LossPhase.GenAndDiscr)

    def __str__(self) -> str:
        return self.value

#----------------------------------------------------------------------------
# Configuration for checkpoint loading.
@dataclass(frozen=True)
class SnapshotConfig:
    # Options for finding the snapshot path.
    snapshot_path: str = None
    snapshot_dir: str = None
    snapshot_exp_id: str = None
    selection_metric: str = None
    snapshot_iter: int = None

    exp_cfg_path: str = None
    cfg_overwrites: dict = None
    use_ema: bool = True
    load_state: bool = True
    allow_missing_keys: bool = False
    allow_unexpected_keys: bool = False
    fsdp: bool = False
    convert_params_to_buffers: bool = False
    graceful_termination_when_missing: bool = False
    external: bool = False
    experiments_dir: str = 'experiments'

#----------------------------------------------------------------------------

