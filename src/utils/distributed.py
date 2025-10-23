import os
import json
import logging
from loguru import logger
import datetime
import warnings
import itertools
from dataclasses import dataclass
from typing import Any
from contextlib import contextmanager

import numpy as np
import torch

#----------------------------------------------------------------------------
# Initialization utils.

def init(disable_c10d_logging: bool=True, timeout_seconds: int=None):
    if disable_c10d_logging:
        logging.getLogger('torch.distributed.distributed_c10d').setLevel(logging.WARNING)

    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '29500'
    if 'RANK' not in os.environ:
        os.environ['RANK'] = '0'
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = '0'
    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = '1'

    backend = 'gloo' if os.name == 'nt' else 'nccl'
    timeout_kwargs = {} if timeout_seconds is None else dict(timeout=datetime.timedelta(seconds=timeout_seconds))
    torch.distributed.init_process_group(backend=backend, init_method='env://', **timeout_kwargs)
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))


def init_random_state_and_cuda(seed: int, cudnn_benchmark: bool, allow_tf32: bool=False):
    # TODO: why is it here?
    np.random.seed((seed * get_world_size() + get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.set_float32_matmul_precision('high')

#----------------------------------------------------------------------------
# Infra distributed utils.

def get_rank():
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

def get_local_rank():
    return int(os.environ['LOCAL_RANK']) if torch.distributed.is_initialized() and 'LOCAL_RANK' in os.environ else 0

def is_main_process():
    return get_rank() == 0

def is_local_main_process():
    return get_local_rank() == 0

def get_world_size():
    return torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

def get_num_nodes() -> int:
    # Assuming that we have the same number of GPUs per node.
    return get_world_size() // max(torch.cuda.device_count(), 1)

def info0(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)

def print0(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)

def warn0(*args, **kwargs):
    if is_main_process():
        warnings.warn(*args, **kwargs)

def loginfo0(*args, **kwargs):
    if is_main_process():
        logger.info(*args, **kwargs)

def logwarn0(*args, **kwargs):
    if is_main_process():
        logger.warning(*args, **kwargs)

def sync_print(*args, **kwargs):
    # Prints stuff on each rank ony by one in a synchronized manner. Very slow, but useful for debugging.
    for rank in range(get_world_size()):
        if rank == get_rank():
            print(*args, **kwargs)
        barrier()

def barrier(*args, **kwargs) -> Any:
    return torch.distributed.barrier(*args, **kwargs) if torch.distributed.is_initialized() else None

def sync_barrier() -> None:
    # Synchronizes all processes by sharing a dummy tensor.
    # A normal barrier would fail to do so since some processes might be stuck in the previous one.
    dummy = torch.tensor([0], device=torch.device('cuda'))
    torch.distributed.all_reduce(dummy, op=torch.distributed.ReduceOp.SUM)

def destroy_process_group():
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

@contextmanager
def leader_first(local: bool=False):
    if get_world_size() == 1:
        yield
    else:
        assert torch.distributed.is_available(), 'Cannot use leader_first() without torch.distributed being available.'
        assert torch.distributed.is_initialized(), 'Cannot use leader_first() without torch.distributed being initialized.'
        if is_main_process() or (local and is_local_main_process()):
            yield
            torch.distributed.barrier()
        else:
            torch.distributed.barrier()
            yield

def maybe_sync_state(net: torch.nn.Module):
    """Syncs parameters and buffers from rank=0 to everyone else"""
    if get_world_size() == 1:
        return

    for tensor in itertools.chain(net.parameters(), net.buffers()):
        safe_broadcast(tensor)

def safe_broadcast(tensor: torch.Tensor):
    if torch.distributed.get_backend() == torch.distributed.Backend.NCCL:
        # NCCL backend requires tensors to be on GPU
        if tensor.device.type != 'cuda':
            tensor_data_cuda = tensor.data.cuda()
            torch.distributed.broadcast(tensor_data_cuda, src=0)
            tensor.data.copy_(tensor_data_cuda.cpu())
        else:
            torch.distributed.broadcast(tensor.data, src=0)
    else:
        # Gloo backend can handle CPU tensors directly
        torch.distributed.broadcast(tensor.data, src=0)

#----------------------------------------------------------------------------
# Aggregation utils.

def mean_across_gpus(x: torch.Tensor) -> torch.Tensor:
    assert str(x.device) != 'cpu', f"Cannot reduce tensor on CPU: {x.device}"
    torch.distributed.reduce(x, dst=0, op=torch.distributed.ReduceOp.SUM)
    return x / get_world_size()

def gather_across_gpus(x: torch.Tensor, group=None) -> torch.Tensor:
    assert str(x.device) != 'cpu', f"Cannot gather tensor on CPU: {x.device}"
    world_size = get_world_size() if group is None else group.size()
    x_out = torch.empty_like(x).unsqueeze(0).repeat_interleave(world_size, dim=0) # [world_size, *x_shape]
    torch.distributed.all_gather_into_tensor(x_out, x)
    return x_out

def world_round(n: int, group=None) -> int:
    """Rounds a number `n` to the nearest larger multiple of the world size."""
    world_size = get_world_size() if group is None else group.size()
    return ((n + world_size - 1) // world_size) * world_size

def gather_seeded_results(x: torch.Tensor, seeds: torch.Tensor, group=None, should_sort: bool=True, drop_duplicates: bool=True, cpu_offload_dims: list[int]=None) -> torch.Tensor | dict:
    """
    We assume that `x` was computed on each GPU using seeds.
    """
    if isinstance(x, dict):
        keys = sorted(list(x.keys()))
        return {k: gather_seeded_results(x[k], seeds, group=group, should_sort=should_sort, drop_duplicates=drop_duplicates, cpu_offload_dims=cpu_offload_dims) for k in keys}
    assert len(x) == len(seeds), f"Wrong shapes: {x.shape}, {seeds.shape}"
    world_size = get_world_size() if group is None else group.size()

    # Gathering seeds
    seeds_out = torch.empty_like(seeds).unsqueeze(0).repeat_interleave(world_size, dim=0) # [world_size, n, ...]
    torch.distributed.all_gather_into_tensor(seeds_out, seeds, group=group)
    x_out = all_gather_into_tensor_with_offload(x, group=group, cpu_offload_dims=cpu_offload_dims) # [world_size, n, ...]

    x_out = x_out.flatten(0, 1) # [world_size * n, ...]
    seeds_out = seeds_out.flatten(0, 1) # [world_size * n, ...]

    if drop_duplicates:
        assert should_sort, "Dropping duplicates requires sorting seeds"
        _, idx_unique = np.unique(seeds_out.cpu().numpy(), return_index=True)
        x_out = x_out[idx_unique]
        seeds_out = seeds_out[idx_unique]

    if should_sort and not drop_duplicates:
        seeds_out, indices = torch.sort(seeds_out, dim=0)
        x_out = x_out[indices]

    return x_out

def all_gather_into_tensor_with_offload(x: torch.Tensor, group=None, cpu_offload_dims: list[int]=None) -> torch.Tensor:
    world_size = get_world_size() if group is None else group.size() # [1]
    if cpu_offload_dims is not None and len(cpu_offload_dims) > 0:
        # Batch-wise gathering with CPU off-loading for the specified dimensions.
        x_out = []
        shape_orig = x.shape
        dim_permutation_for_offloading = cpu_offload_dims + [i for i in range(len(shape_orig)) if not i in cpu_offload_dims] # We want to put the offloading dims first and flatten them.
        dim_permutation_for_offloading_inverse = [dim_permutation_for_offloading.index(i) for i in range(len(shape_orig))]
        x_reshaped_for_offloading = x.permute(dim_permutation_for_offloading).flatten(0, len(cpu_offload_dims) - 1) # [world_size * n, D, ...]
        x_out_tmp_store = torch.empty_like(x_reshaped_for_offloading[[0]]).unsqueeze(0).repeat_interleave(world_size, dim=0) # [world_size, 1, D, ...]
        for i in range(len(x_reshaped_for_offloading)):
            torch.distributed.all_gather_into_tensor(x_out_tmp_store, x_reshaped_for_offloading[[i]], group=group)
            x_out.append(x_out_tmp_store.cpu())
        x_out = torch.cat(x_out, dim=1) # [world_size, D, ...]

        # Reshaping back to the original shape
        x_out = x_out.reshape(world_size, *[shape_orig[i] for i in cpu_offload_dims], *[shape_orig[i] for i in range(len(shape_orig)) if not i in cpu_offload_dims])
        x_out = x_out.permute(0, *[(i + 1) for i in dim_permutation_for_offloading_inverse]) # [world_size, ...]
    else:
        x_out = torch.empty_like(x).unsqueeze(0).repeat_interleave(world_size, dim=0) # [world_size, n, ...]
        torch.distributed.all_gather_into_tensor(x_out, x, group=group)
    return x_out

def gather_concat(x: torch.Tensor):
    if get_world_size() == 1:
        return x

    x = x.contiguous()
    ys = []
    for src_rank in range(get_world_size()):
        y = x.clone()
        torch.distributed.broadcast(y, src=src_rank)
        ys.append(y)
    x = torch.cat(ys, dim=0) # [w * n, ...]
    return x

def gather_concat_not_nan(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 2, f"Wrong shape: {x.shape}"
    x = gather_concat(x) # [w * n, ...]
    nan_mask = torch.isnan(x).any(dim=1) # [w * n, ...]
    x = x[~nan_mask] # [w * n, ...]
    return x

#----------------------------------------------------------------------------

STATE_PATH = 'state.json' # Path to the state file with a stopping criteria.

@dataclass(frozen=True)
class ControlSignal:
    should_stop: bool = False
    should_snapshot: bool = False
    should_save_samples: bool = False

    def sync(self):
        """Broadcasts the control signal to all processes."""
        signal = torch.tensor([self.should_stop, self.should_snapshot], dtype=torch.bool, device=torch.device('cuda'))
        torch.distributed.broadcast(signal, src=0)
        return ControlSignal(
            should_stop=signal[0].item(),
            should_snapshot=signal[1].item(),
            should_save_samples=self.should_save_samples,
        )

    @staticmethod
    def clear_file_state():
        if is_main_process() and os.path.exists(STATE_PATH):
            os.remove(STATE_PATH)

    @staticmethod
    def receive_from_file() -> "ControlSignal":
        if is_main_process() and os.path.exists(STATE_PATH):
            with open(STATE_PATH, 'r') as f:
                state_dict = json.load(f)
            ControlSignal.clear_file_state()
        else:
            state_dict = {}
        return ControlSignal(
            should_stop=state_dict.get('should_stop', False),
            should_snapshot=state_dict.get('should_snapshot', False),
            should_save_samples=state_dict.get('should_save_samples', False),
        )

    @property
    def should_do_something(self) -> bool:
        return any([self.should_stop, self.should_snapshot, self.should_save_samples])

#----------------------------------------------------------------------------
