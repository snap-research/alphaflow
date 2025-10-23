import os
import sys
import re
import copy
import json
import traceback
from typing import Union, Any, Optional
from dataclasses import asdict, replace

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp.api import FullOptimStateDictConfig
from omegaconf import OmegaConf, DictConfig

try:
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
except:
    fully_shard = MixedPrecisionPolicy = None

from src.structs import EasyDict, TensorGroup
from src.utils import misc, distributed as dist, config_utils, logging_utils, os_utils
from src.utils.os_utils import construct_class_by_name
from src.structs import Nets, Optimizers, ResumeConfig, SnapshotConfig, DTYPE_STRING_TO_DTYPE, FSDP_STRATEGY_STRING_TO_FSDP_STRATEGY
from src.data import Data

#----------------------------------------------------------------------------
# Network/optimizer/etc initialization functions.

def init_modules(cfg, device, data: Data, start_time, resume_cfg: ResumeConfig, load_data: bool=True, load_vis: bool=True):
    nets: Nets = init_networks(cfg, device)
    _, loss_kwargs = instantiate_network_kwargs(cfg)

    # Make sure that the optimizer is created after the FSDP model.
    dist.loginfo0('Setting up the network and optimizer... ', end='')
    loss_fn = construct_class_by_name(**loss_kwargs).to(device) # training.loss.(VP|VE|EDM)Loss
    optimizers: Optimizers = init_optimizers(cfg, nets)
    dist.loginfo0('Done!')
    if not cfg.model.always_weight_decay:
        dist.loginfo0('decayable params:\n', '\n'.join([n for n, p in nets.net.named_parameters() if p.requires_grad and not getattr(p, 'no_weight_decay', False)]))
        dist.loginfo0('non-decayable params:\n', '\n'.join([n for n, p in nets.net.named_parameters() if p.requires_grad and getattr(p, 'no_weight_decay', False)]))

    # Resume training from previous snapshot (when available and asked).
    dist.loginfo0('Resuming from snapshot (if applicable)... ', end='')
    snapshot = load_states(cfg, nets, optimizers, resume_cfg, data=data if load_data else None)
    stats = logging_utils.TrainingStats.init_from_snapshot(snapshot, start_time=start_time, resume_whole_state=resume_cfg.whole_state, allow_missing_state=resume_cfg.allow_missing_stats)
    vis = (EasyDict.init_recursively(snapshot['vis']) if (resume_cfg.whole_state and snapshot is not None and 'vis' in snapshot) else logging_utils.load_vis_data(cfg, loss_fn, data.dataset)) if load_vis else None

    del snapshot; misc.free_memory() # Delete resume data from memory
    dist.loginfo0('Done!')

    return nets, optimizers, loss_fn, vis, stats


def init_networks(cfg, device) -> Nets:
    dist.loginfo0('Constructing network... ')
    interface_kwargs = dict(resolution=cfg.dataset.resolution, img_channels=3, label_shape=cfg.dataset.label_shape)
    network_kwargs, _ = instantiate_network_kwargs(cfg)

    net: torch.nn.Module = construct_class_by_name(**network_kwargs, **interface_kwargs)
    dist.maybe_sync_state(net)
    net.cpu().train().requires_grad_(True).to(dtype=DTYPE_STRING_TO_DTYPE[cfg.model.param_dtype])
    if cfg.model.is_lgm and hasattr(net, 'autoencoder') and isinstance(net.autoencoder, torch.nn.Module):
        net.autoencoder.eval().requires_grad_(False)
    ema = None


    assert cfg.training.dp.strategy in ('ddp', 'fsdp', 'fsdp2', None), f'Unsupported dp strategy: {cfg.training.dp.strategy}'

    if cfg.model.use_ema and cfg.training.dp.strategy in ('fsdp', 'fsdp2'):
        # For FSDP, we need to create EMA right after the raw model creation, since FSDP creates some unpickable process group.
        ema = copy.deepcopy(net).eval().requires_grad_(False)
    num_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad) # (1)
    net_dp = init_net_dp(cfg.training.dp, net, device, is_trainable=True)
    if cfg.model.use_ema and cfg.training.dp.strategy in ('ddp', None):
        # For DDP, we need to create it after the DDP model is created, since we can't create DDP for EMA (it's not trainable), and the states across ranks will be unsynced.
        ema = copy.deepcopy(net).eval().requires_grad_(False)
    ema_dp = init_net_dp(cfg.training.dp, ema, device, is_trainable=False) if ema is not None else None

    if cfg.model.compile:
        assert cfg.training.dp.strategy in ('ddp', None), f'Compilation is only supported for DDP, but got {cfg.training.dp.strategy}'
        net_dp = torch.compile(net_dp)
        ema_dp = torch.compile(ema_dp) if ema_dp is not None else None

    dist.loginfo0(f'Number of trainable parameters: {num_trainable_params:,}')

    return Nets(
        net=net_dp.to(device),
        ema=ema_dp.to(device) if ema_dp is not None else None,
    )


def init_optimizers(cfg: DictConfig, nets: Nets) -> Optimizers:
    if cfg.training.force_single_group_params:
        # This branch is needed solely for compatibility with the old HPDM checkpoints. TODO: remove this legacy.
        assert cfg.model.always_weight_decay, 'force_single_group_params requires always_weight_decay=true'
        params = nets.net.parameters()
    else:
        params = [
            # Params that always have weight decay.
            {'params': [p for p in nets.net.parameters() if p.requires_grad and not getattr(p, 'no_weight_decay', False)]},
            # Params that don't have weight decay (unless `cfg.model.always_weight_decay` is set to True). We always split them into two groups to be able to fine-tune various snapshots (otherwise, they might have different grouping).
            {'params': [p for p in nets.net.parameters() if p.requires_grad and getattr(p, 'no_weight_decay', False)], 'weight_decay': 0.0 if not cfg.model.always_weight_decay else cfg.model.optim.weight_decay},
        ]

    optim_common_kwargs = dict(class_name=cfg.model.optim.class_name, params=params, lr=cfg.model.optim.lr)
    if cfg.model.optim.class_name in ('torch.optim.AdamW', 'torch.optim.Adam'):
        optim_extra_kwargs = dict(betas=cfg.model.optim.betas, eps=cfg.model.optim.eps, weight_decay=cfg.model.optim.weight_decay)
    elif cfg.model.optim.class_name == 'torch.optim.SGD':
        optim_extra_kwargs = dict(momentum=cfg.model.optim.momentum, weight_decay=cfg.model.optim.weight_decay)
    else:
        raise NotImplementedError(f'Unsupported optimizer: {cfg.model.optim.class_name}')
    optimizer = construct_class_by_name(**optim_common_kwargs, **optim_extra_kwargs) # subclass of torch.optim.Optimizer
    lr_scheduler = construct_class_by_name(optimizer=optimizer, class_name=cfg.model.lr_scheduler.class_name, cfg=cfg.model.lr_scheduler) # subclass of torch.optim.lr_scheduler.LRScheduler
    grad_scaler = (ShardedGradScaler() if cfg.training.dp.strategy == 'fsdp' else torch.amp.GradScaler()) if cfg.model.autocast and cfg.model.use_autocast_grad_scaler else None # pylint: disable=protected-access
    return Optimizers(
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        grad_scaler=grad_scaler,
    )


def instantiate_network_from_cfg(experiment_cfg: Union[EasyDict, dict[str, Any]]) -> torch.nn.Module:
    network_kwargs, _ = instantiate_network_kwargs(experiment_cfg)
    interface_kwargs = dict(resolution=experiment_cfg.dataset.resolution, img_channels=3, label_shape=experiment_cfg.dataset.label_shape)
    net = construct_class_by_name(**network_kwargs, **interface_kwargs)
    if not hasattr(net, 'cfg'):
        net.cfg = experiment_cfg.model
    return net


def instantiate_network_kwargs(experiment_cfg: Union[DictConfig, EasyDict]):
    network_kwargs = EasyDict(class_name=experiment_cfg.model.target_class, cfg=experiment_cfg.model)
    loss_kwargs = EasyDict(class_name=experiment_cfg.loss.target, cfg=experiment_cfg.loss)

    return network_kwargs, loss_kwargs

#----------------------------------------------------------------------------
# Distributed training initialization functions.

def init_net_dp(dp_strategy_cfg: EasyDict, net: torch.nn.Module, device: torch.device, is_trainable: bool=True) -> torch.nn.Module:
    if dp_strategy_cfg.strategy == 'ddp' and is_trainable:
        # A non-trainable DDP model would raise an error.
        net_dp = torch.nn.parallel.DistributedDataParallel(net.to(device), device_ids=[device], broadcast_buffers=False, find_unused_parameters=dp_strategy_cfg.ddp_find_unused_parameters)
    elif dp_strategy_cfg.strategy == 'fsdp':
        net.to(device) # Put on the device before wrapping with FSDP to make sure that all the params are on the same device.
        net_dp = init_net_fsdp(net, dp_strategy_cfg.fsdp_min_num_params, fsdp_strategy=dp_strategy_cfg.fsdp_strategy)
    elif dp_strategy_cfg.strategy == 'fsdp2':
        net_dp = init_net_fsdp2(net)
    else:
        net_dp = net

    return net_dp


def init_net_fsdp(net: torch.nn.Module, fsdp_min_num_params: int=100_000_000_000, fsdp_strategy: str='full_shard') -> torch.nn.Module:
    auto_wrap_policy = net.get_autowrap_policy(fsdp_min_num_params)
    # TODO: try forward_prefetch=True
    net_dp = FSDP(net, auto_wrap_policy=auto_wrap_policy, use_orig_params=True, limit_all_gathers=True, sharding_strategy=FSDP_STRATEGY_STRING_TO_FSDP_STRATEGY[fsdp_strategy])

    FSDP.set_state_dict_type(
        net_dp,
        StateDictType.FULL_STATE_DICT,

        # Offloading to CPU for the rank0 only to save both GPU and CPU memory (as recommended in the docs)
        FullStateDictConfig(offload_to_cpu=dist.get_world_size() > 1, rank0_only=True),
        FullOptimStateDictConfig(offload_to_cpu=dist.get_world_size() > 1, rank0_only=True),
    )

    return net_dp

def init_net_fsdp2(net: torch.nn.Module, param_dtype: str=None, reduce_dtype: str=None) -> torch.nn.Module:
    for module in net.children():
        if hasattr(module, 'should_fsdp_wrap') and module.should_fsdp_wrap:
            init_net_fsdp2(module, param_dtype, reduce_dtype)

    fully_shard(
        net,
        reshard_after_forward=False,
        mp_policy=MixedPrecisionPolicy(
            param_dtype=None if param_dtype is None else DTYPE_STRING_TO_DTYPE[param_dtype],
            reduce_dtype=None if reduce_dtype is None else DTYPE_STRING_TO_DTYPE[reduce_dtype],
        ),
    )

    return net

#----------------------------------------------------------------------------
# Snapshot saving, loading and miscellaneous functions.

def load_states(cfg, nets: Nets, optimizers: Optimizers, resume_cfg: ResumeConfig, data: Optional[Data]=None) -> Union[None, dict[str, Any]]:
    if resume_cfg.ckpt_path is None:
        return None

    # Load the snapshot
    dist.loginfo0(f'Resuming from "{resume_cfg.ckpt_path}"')
    misc.free_memory()
    if not dist.is_main_process():
        torch.distributed.barrier() # rank 0 goes first
    with os_utils.open_path(resume_cfg.ckpt_path, verbose=dist.is_main_process()) as f:
        snapshot = torch.load(f, map_location='cpu', weights_only=False) # Mapping to CPU to not keep 2 snapshots in GPU memory
    if dist.is_main_process():
        torch.distributed.barrier() # other ranks follow

    # Load the state dicts.
    modules_to_load = {'net': nets.net, 'ema': nets.ema}
    allowed_to_skip = {'ema': resume_cfg.allow_missing_ema}
    for name, module in modules_to_load.items():
        if module is None:
            continue
        if snapshot.get(name) is None and allowed_to_skip.get(name, False):
            dist.logwarn0(f'Expected to find "{name}" in the snapshot, but got {snapshot.keys()} instead. But skipping since allowed to skip.')
            continue
        else:
            assert name in snapshot, f'Expected to find "{name}" in the snapshot, but got {snapshot.keys()} instead.'
        allow_missing_keys_regex=r"^(autoencoder|are_ldm_stats_inited)" if (name in ['net', 'ema'] and cfg.model.is_lgm) else None
        load_module_state(cfg, module, snapshot, name, resume_cfg=replace(resume_cfg, allow_missing_keys_regex=allow_missing_keys_regex))

    if resume_cfg.whole_state:
        # Convert the optim state dict to the sharded one if needed.
        if snapshot.get('optimizer') is not None:
            optim_state_dict = snapshot['optimizer']
            if cfg.training.dp.strategy == 'fsdp':
                if snapshot.get('cfg', {}).get('training', {}).get('dp_strategy') == 'ddp':
                    assert resume_cfg.allow_missing_optim, 'Expected to load an FSDP optimizer, but got a DDP one instead, and skipping is not allowed.'
                    dist.logwarn0("We do not know how to load a DDP optimizer into FSDP, but allowed to skip, so skipping.")
                    optim_state_dict = None
                else:
                    if resume_cfg.allow_missing_keys:
                        patch_fsdp_optim_state_dict_with_missing_params_(optim_state_dict, [k.replace('_fsdp_wrapped_module.', '') for k, p in nets.net.named_parameters() if p.requires_grad])
                    optim_state_dict = FSDP.optim_state_dict_to_load(optim_state_dict=optim_state_dict, model=nets.net, optim=optimizers.optimizer)
            if optim_state_dict is not None:
                optimizers.optimizer.load_state_dict(optim_state_dict)
        else:
            assert resume_cfg.allow_missing_optim, f'Expected to find "optimizer" in the snapshot, but got {snapshot.keys()} instead.'
        if snapshot.get('lr_scheduler') is not None and optimizers.lr_scheduler is not None:
            optimizers.lr_scheduler.load_state_dict(snapshot['lr_scheduler'])
        if snapshot.get('grad_scaler') is not None and optimizers.grad_scaler is not None:
            optimizers.grad_scaler.load_state_dict(snapshot['grad_scaler'])

        if snapshot.get('dataloader') is not None and data is not None and hasattr(data.dataloader, 'load_state_dict'):
            try:
                data.dataloader.load_state_dict(snapshot['dataloader'])
            except Exception as e:
                dist.logwarn0(f'Failed to load dataloader state dict: {e}. Continuing training without restoring the dataloader state.')
                traceback.print_exc()

    return snapshot


def load_module_state(cfg, module: torch.nn.Module, snapshot, name: str, resume_cfg: ResumeConfig, **load_state_dict_kwargs):
    """
    Loads state parameters/buffers into the model. Potentially re-inits some of them.
    """
    if cfg.training.resume.resize_tokens_posenc:
        tokens_posenc_keys = [k for k in snapshot[name].keys() if any([k.endswith(s) for s in ['tokens_posenc', 'tokens_posenc_vert']])]
        for k in tokens_posenc_keys:
            snapshot[name][k] = misc.unwrap_module(module).model.resize_tokens_posenc(snapshot[name][k], k, snapshot['cfg'])

    incompatible_keys = misc.load_state_dict_(module, snapshot[name], resume_cfg, module_log_name=name, **load_state_dict_kwargs)
    if cfg.training.resume.reinit_new_params:
        params_to_reinit = set(k.replace('_fsdp_wrapped_module.', '') for k in incompatible_keys.missing_keys)
        reinited_new_params = set()
        with misc.summon_full_params(module):
            for param_name, param in misc.named_params_and_buffers(module):
                if param_name in params_to_reinit:
                    misc.unwrap_module(module).model.reinit_new_param_(param_name, param)
                    reinited_new_params.add(param_name)
        assert len(reinited_new_params) == len(params_to_reinit), \
            f'Failed to zero out {len(params_to_reinit - reinited_new_params)}' \
            f'parameters in module {name}: {params_to_reinit - reinited_new_params}'

    dist.loginfo0(f'Loaded {name}. Num missing keys: {len(incompatible_keys.missing_keys)}. Num unexpected keys: {len(incompatible_keys.unexpected_keys)}.')


def save_snapshot(cfg, nets: Nets, optimizers: Optimizers, data: Data, stats: logging_utils.TrainingStats, vis, snapshot_path, reason: str='', save_non_empty_optim: bool=True) -> str:
    assert isinstance(stats, logging_utils.TrainingStats), f'Expected stats to be of type TrainingStats, but got {type(stats)}'
    # Save network snapshot.
    dist.loginfo0(f'Saving the snapshot (reason: {reason})... ', end='')
    dist.barrier() # Having a barrier here to make sure that all the ranks have arrived. If not all the ranks have arrived, the subsequent errors would be cryptic.
    # TODO: check DDP consistency of the optimizer as well
    is_consistent = {}
    for module_name, module in dict(net=nets.net, ema=nets.ema).items():
        if isinstance(module, torch.nn.Module):
            # TODO: does it make sense to check DDP consistency for an FSDP model?
            try:
                misc.check_ddp_consistency(module, prefix=f'[{module_name}] ', ignore_regex = r'.*singular.*|.*(running_mean|running_var|num_batches_tracked)$')
                is_consistent[module_name] = True
            except Exception as e: # pylint: disable=broad-except
                dist.logwarn0(f'DDP consistency check failed for {module_name}: {e}')
                traceback.print_exc()
                is_consistent[module_name] = False
    dist.barrier()

    if save_non_empty_optim:
        if cfg.training.dp.strategy == 'fsdp':
            optim_state_dict = FSDP.optim_state_dict(model=nets.net, optim=optimizers.optimizer) # Gather the state dict from all the ranks.
            # optim_state_dict = FSDP.rekey_optim_state_dict(optim_state_dict, OptimStateKeyType.PARAM_NAME, net) # Rekey to the universal format.
        else:
            optim_state_dict = optimizers.optimizer.state_dict()
        if not dist.is_main_process() or len(optim_state_dict['state']) == 0:
            optim_state_dict = None # Do not pack the optimizer if it is empty.
    else:
        optim_state_dict = None

    snapshot = {
        # Main module's state.
        'net': misc.unwrap_module(nets.net).state_dict(),
        'ema': misc.unwrap_module(nets.ema).state_dict() if nets.ema is not None else None,
        'lr_scheduler': None if optimizers.lr_scheduler is None else optimizers.lr_scheduler.state_dict(),
        'grad_scaler': None if optimizers.grad_scaler is None else optimizers.grad_scaler.state_dict(),
        'optimizer': optim_state_dict, # We'll fill the optimizer later.

        # Miscellanous states.
        'cfg': cfg.to_dict(),
        'vis': {k: (v.to('cpu') if isinstance(v, (torch.Tensor, TensorGroup)) else v) for k, v in vis.items()},
        'stats': asdict(stats),
        'dataloader': data.dataloader.state_dict() if hasattr(data.dataloader, 'state_dict') else None, # not all dataloaders have a state dict
        'ddp_consistency_checks': is_consistent,
    }
    if dist.is_main_process():
        snapshot_path_tmp = snapshot_path.replace(os.path.basename(snapshot_path), 'snapshot-tmp.pt')
        torch.save(snapshot, snapshot_path_tmp)
        os.replace(snapshot_path_tmp, snapshot_path)
    dist.loginfo0('Saved!')
    dist.barrier() # Making sure that some processes have not exited (when we are done training).
    del snapshot; misc.free_memory() # conserve memory


def get_snapshot_path(cfg, stats, snapshot_suffix=None) -> os.PathLike:
    snapshot_suffix = f"{stats.cur_step:08d}" if snapshot_suffix is None else snapshot_suffix
    return os.path.join(cfg.output_dir, f'snapshot-{snapshot_suffix}.pt')


def patch_fsdp_optim_state_dict_with_missing_params_(state_dict_to_load: dict, all_param_names: list[str]) -> dict:
    dist.loginfo0('Loading optimizer state dict with missing params...')
    cur_params = state_dict_to_load['param_groups'][0]['params']
    cur_state = state_dict_to_load['state']

    if dist.is_main_process(): # Running checks in the main process only for easier printing.
        assert all(p in cur_state for p in cur_params), f"Expected all params from `params` to be in `state`, but got missing: {set(cur_params) - set(cur_state)}"
        assert all(p in cur_params for p in cur_state), f"Expected all params from `state` to be in `params`, but got missing: {set(cur_state) - set(cur_params)}"
        assert len(all_param_names) >= len(cur_params), f"Expected {len(all_param_names)} >= {len(cur_params)}"
        assert all(p in all_param_names for p in cur_params), f"Expected all params from `params` to be in `all_param_names`, but got: \nfull_params: {all_param_names}, \ncur_params: {cur_params}"
        assert len(state_dict_to_load['param_groups']) == 1, f"Expected 1 param group, but got {len(state_dict_to_load['param_groups'])}"
        # assert len(full_state_dict['param_groups']) == 1, f"Expected 1 param group, but got {len(full_state_dict['param_groups'])}"
        assert len(cur_params) == len(state_dict_to_load['state']), f"Expected {len(cur_params)} == {len(state_dict_to_load['state'])}"
        assert all(isinstance(p, str) for p in cur_params), f"Expected all current params to be strings (for FSDP), but got non-string params: {[p for p in cur_params if not isinstance(p, str)]}"
        assert all(isinstance(p, str) for p in all_param_names), f"Expected all params to be strings (for FSDP), but got non-string params: {[p for p in all_param_names if not isinstance(p, str)]}"

    missing_params = {p: i for i, p in enumerate(all_param_names) if not p in state_dict_to_load['param_groups'][0]['params']}
    dist.loginfo0('Missing optim params:', missing_params)
    offset = 0
    for param, i in missing_params.items():
        cur_state[param] = {} # We do not need to do state initialization, because it will be done by the optimizer.
        cur_params.insert(i + offset, param)
        offset += 1

#----------------------------------------------------------------------------
# Snapshot loading functions.

def infer_snapshot_path(snapshot_cfg: SnapshotConfig, verbose: bool=True) -> os.PathLike:
    if snapshot_cfg.snapshot_exp_id is not None:
        assert snapshot_cfg.snapshot_dir is None and snapshot_cfg.snapshot_path is None, "if snapshot_exp_id is provided, snapshot_dir and snapshot_path should be None"
        assert os.path.isdir(snapshot_cfg.experiments_dir), "`experiments` directory should exist"
        matched_exp_dirs = [d for d in os.listdir(snapshot_cfg.experiments_dir) if re.match(f'{snapshot_cfg.snapshot_exp_id}-.*', d)]
        assert len(matched_exp_dirs) == 1, f"Expected to find exactly one experiment directory with the ID {snapshot_cfg.snapshot_exp_id}, but got {matched_exp_dirs}"
        snapshot_dir = os.path.join(snapshot_cfg.experiments_dir, matched_exp_dirs[0], 'output')
    else:
        snapshot_dir = snapshot_cfg.snapshot_dir

    if snapshot_dir is not None:
        assert snapshot_cfg.snapshot_path is None, "if snapshot_dir is provided, snapshot_path and snapshot_exp_id should be None"
        if snapshot_cfg.selection_metric is not None:
            metrics_file = os.path.join(snapshot_dir, f'metric-{snapshot_cfg.selection_metric}.jsonl')
            with open(metrics_file, 'r') as f:
                snapshot_metrics_vals = [json.loads(line) for line in f.read().splitlines()]
            snapshot = sorted(snapshot_metrics_vals, key=lambda m: m['results'][snapshot_cfg.selection_metric])[0]
            snapshot_path = os.path.join(snapshot_dir, snapshot['snapshot_path'])
            if verbose:
                print(f'Using checkpoint: {snapshot_path} with {snapshot_cfg.selection_metric} of', snapshot['results'][snapshot_cfg.selection_metric])
        elif snapshot_cfg.snapshot_iter is not None:
            snapshot_path = os.path.join(snapshot_dir, f'snapshot-{snapshot_cfg.snapshot_iter:08d}.pt')
            if verbose:
                print(f"Loading the requested via iter snapshot: {snapshot_path}")
        else:
            ckpt_regex = re.compile(r"^snapshot-\d{8}.pt$")
            ckpts = sorted([f for f in os.listdir(snapshot_dir) if ckpt_regex.match(f)])
            snapshot_path = os.path.join(snapshot_dir, ckpts[-1])
            if verbose:
                print(f"Using the latest found checkpoint: {snapshot_path}")
    elif snapshot_cfg.snapshot_path is not None:
        assert snapshot_dir is None, "if snapshot_path is provided, snapshot_dir and snapshot_exp_id should be None"
        snapshot_path = snapshot_cfg.snapshot_path
    else:
        raise NotImplementedError('Must provide either snapshot_dir or snapshot_path.')

    return snapshot_path


def load_snapshot(snapshot_cfg: DictConfig, verbose: bool=True, device='cpu'):
    snapshot_cfg = SnapshotConfig(**snapshot_cfg) if not isinstance(snapshot_cfg, SnapshotConfig) else snapshot_cfg
    snapshot_path = infer_snapshot_path(snapshot_cfg, verbose=verbose)

    # Load network.
    if not os.path.isfile(snapshot_path):
        error_msg = f'--network must point to a file, but got {snapshot_path}'
        if snapshot_cfg.graceful_termination_when_missing:
            dist.loginfo0(f'{error_msg}. Gracefully terminating since graceful_termination_when_missing is set to True.')
            sys.exit(0)
        else:
            raise FileNotFoundError(error_msg)

    if verbose:
        dist.loginfo0(f'Loading the snapshot from {snapshot_path}')

    # TODO: pass the cuda device id to load faster.
    snapshot = torch.load(snapshot_path, map_location='cpu', weights_only=False)
    experiment_cfg = snapshot['cfg'] if snapshot_cfg.exp_cfg_path is None else OmegaConf.load(snapshot_cfg.exp_cfg_path)
    if snapshot_cfg.cfg_overwrites is not None and len(snapshot_cfg.cfg_overwrites) > 0:
        dist.loginfo0(f'Overwriting the config with: {snapshot_cfg.cfg_overwrites}')
        experiment_cfg = experiment_cfg.to_dict() if isinstance(experiment_cfg, EasyDict) else experiment_cfg # TODO: remove this legacy; make sure that the config is always a dict...
        experiment_cfg = OmegaConf.merge(experiment_cfg, EasyDict.init_recursively(snapshot_cfg.cfg_overwrites).to_dict())
    experiment_cfg = config_utils.legacy_patch_experiment_cfg(EasyDict.init_recursively(experiment_cfg))
    net = instantiate_network_from_cfg(experiment_cfg).to(device=device, dtype=DTYPE_STRING_TO_DTYPE[experiment_cfg.model.param_dtype])
    if snapshot_cfg.convert_params_to_buffers:
        misc.replace_parameters_with_buffers_(net)
    if snapshot_cfg.fsdp:
        assert str(device) != 'cpu', "FSDP is not supported on CPU."
        if verbose:
            dist.loginfo0('Initializing FSDP...')
        net = init_net_fsdp(net, fsdp_strategy=snapshot_cfg.training.dp.fsdp_strategy)
    if snapshot_cfg.load_state:
        module_name = 'ema' if snapshot_cfg.use_ema else 'net'
        if verbose:
            dist.loginfo0(f'Loading the state dict for {module_name}')
        resume_cfg = ResumeConfig(
            ckpt_path=snapshot_path, whole_state=False, allow_missing_keys=snapshot_cfg.allow_missing_keys,
            allow_unexpected_keys=snapshot_cfg.allow_unexpected_keys)
        incompatible_keys = misc.load_state_dict_(net, snapshot[module_name], resume_cfg, module_log_name=module_name)
        if incompatible_keys:
            dist.loginfo0(f'Incompatible keys: {incompatible_keys}')

    return net, snapshot_path, experiment_cfg

#----------------------------------------------------------------------------
