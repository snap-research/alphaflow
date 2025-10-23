"""
This script computes imgs/sec for a generator for different batch sizes
"""
# import debugpy, os
# # debugpy.listen(("127.0.0.1", 5678))
# # print("Waiting for debugger…"); debugpy.wait_for_client()
# debugpy.breakpoint()

import sys; sys.path.extend(['..', '.', 'src'])
import time
import contextlib
import itertools

import numpy as np
import torch
import torch.profiler as profiler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from omegaconf import DictConfig
from tqdm import tqdm

from src.structs import Nets, EasyDict, TensorGroup
from src.utils import misc, logging_utils, distributed as dist
from src.training.network_utils import init_networks
from src.train import update_ema
from infra.utils import recursive_instantiate
from infra.launch import launch_from_registry

#----------------------------------------------------------------------------

def profile_for_batch_size(nets: Nets, cfg: DictConfig, batch_size: int, resolution: tuple[int], run_backward: bool=False, profiling_row_limit: int=10, full_summary: bool=False, optim=None, device=None):
    net_bare = misc.unwrap_module(nets.net)

    x, sigma, cond = net_bare.construct_dummy_input(batch_size, device=device)
    x_ae, sigma_ae, cond_ae = net_bare.autoencoder.construct_dummy_input(batch_size, device=device) if cfg.model.is_lgm and not cfg.get('lgm_only', False) else (None, None, None)
    times = []

    if cfg.get('verbose', True):
        logging_utils.print_model_info(cfg, batch_size, nets, device)

    time.sleep(1)
    torch.cuda.reset_peak_memory_stats()
    num_warmup_iters = cfg.get('num_warmup_iters', 5)
    profiler_ctx = contextlib.nullcontext() if not full_summary else profiler.profile(record_shapes=True, use_cuda=True)

    with profiler_ctx as prof:
        for _i in tqdm(range(num_warmup_iters + cfg.get('num_profile_iters', 25)), desc='Profiling'):
            torch.cuda.synchronize()
            start_time = time.time()

            with profiler.record_function("forward"): # Using autocast only for forward (as they recommend)
                if x_ae is not None:
                    with torch.no_grad():
                        x = misc.unwrap_module(nets.net).encode(x_ae, sigma=sigma_ae, cond=cond_ae)

                if cfg.get('encode_only', False):
                    x_out = nets.net(x, sigma=sigma, cond=cond, decode=False, return_extra_output=True)[1]['latents'].contiguous()
                else:
                    x_out = nets.net(x, sigma=sigma, cond=cond)
                    x_out = TensorGroup(x_out) if isinstance(x_out, dict) else x_out
                    x_out = x_out.contiguous()

                if not run_backward:
                    _ = x_out.view(-1)[0].item() # TODO: this also includes the speed of moving from GPU to CPU.
                    torch.cuda.synchronize()
                    times.append(time.time() - start_time)

            if run_backward:
                with profiler.record_function("backward"):
                    loss = x_out.reduce_sum() if isinstance(x_out, TensorGroup) else x_out.sum()
                    # loss = loss_fn(net=net, videos=noise, cond=cond)
                    loss.backward()
                    if not cfg.model.grad_clip.norm is None:
                        _norm = torch.nn.utils.clip_grad_norm_(nets.net.parameters(), cfg.model.grad_clip.norm) # [1]
                    torch.cuda.synchronize()
                    times.append(time.time() - start_time)

            if not optim is None:
                optim.step()

            if nets.ema is not None and cfg.get('update_ema', True):
                update_ema(cfg, nets.net, nets.ema, stats=EasyDict(cur_step=12345))

    times = times[num_warmup_iters:] # Removing warmup iterations
    torch.cuda.empty_cache()
    num_imgs_processed = len(times) * batch_size
    total_time_spent = np.sum(times)
    samples_per_sec = num_imgs_processed / total_time_spent

    if full_summary:
        summary = prof.key_averages().table(sort_by="cpu_time_total", row_limit=profiling_row_limit)
    else:
        summary = None

    dist.print0(f'[Batch size: {batch_size} {resolution}] mean: {np.mean(times):.05f}s/it. std: {np.std(times):.05f}s')
    dist.print0(f'[Batch size: {batch_size} {resolution}] samples/sec: {samples_per_sec:.03f}')
    dist.print0(f'[Batch size: {batch_size} {resolution}] sec/kimg: {1000.0/samples_per_sec:.03f}')
    dist.print0(f'[Batch size: {batch_size} {resolution}] max mem: {torch.cuda.max_memory_allocated(device) / 2**30:<6.2f} gb')

    return samples_per_sec, summary

#----------------------------------------------------------------------------

def profile_model(cfg: DictConfig):
    device = torch.device('cuda')
    recursive_instantiate(cfg)
    cfg = EasyDict.init_recursively(cfg)

    dist.init()
    dist.init_random_state_and_cuda(42, cfg.training.cudnn_benchmark, cfg.training.allow_tf32)
    all_backward = cfg.get('all_backward', True)
    use_grad = cfg.get('use_grad', all_backward)
    run_backward = cfg.get('run_backward', all_backward)
    train_mode = cfg.get('train_mode', all_backward)
    full_summary = cfg.get('full_summary', False)
    save_ckpt_path = cfg.get('save_ckpt_path')
    nets = init_networks(cfg, device)
    nets.net.train(train_mode).requires_grad_(use_grad)

    if cfg.get('compile_mode') is not None:
        dist.loginfo0('Compiling model...')
        net_compiled = torch.compile(nets.net, mode=cfg.get('compile_mode'))
        nets = Nets(net_compiled, nets.ema)

    optim = torch.optim.Adam(nets.net.parameters(), lr=0.0) if cfg.get('run_optim') else None
    if cfg.get('init_only'):
        return
    batch_sizes = [cfg.dataset.batch_gpu] if cfg.get('profile_batch_sizes') is None else cfg.profile_batch_sizes
    bandwidths = []
    summaries = []
    resolutions = [misc.unwrap_module(nets.net).input_resolution] if cfg.get('profile_resolutions') is None else cfg.profile_resolutions

    with torch.set_grad_enabled(mode=use_grad):
        for batch_size, resolution in itertools.product(batch_sizes, resolutions):
            bandwidth, summary = profile_for_batch_size(
                nets=nets,
                cfg=cfg,
                batch_size=batch_size,
                resolution=resolution,
                run_backward=run_backward,
                profiling_row_limit=cfg.get('profiling_row_limit', 10),
                full_summary=full_summary,
                optim=optim,
                device=device,
            )
            bandwidths.append(bandwidth)
            summaries.append(summary)

    if save_ckpt_path is not None:
        dist.loginfo0(f'Saving checkpoint to {save_ckpt_path}...', end=' ')
        state_dict = nets.net.state_dict()
        if dist.is_main_process():
            torch.save(state_dict, save_ckpt_path)
        dist.loginfo0('Done!')

        optim_state_dict = (FSDP.full_optim_state_dict(nets.net, optim) if cfg.get('profile_fsdp') else optim.state_dict()) if optim is not None else None
        if optim_state_dict is not None and dist.is_main_process():
            optim_save_path = save_ckpt_path.replace('.pt', '_optim.pt')
            dist.loginfo0(f'Saving optimizer to {optim_save_path}...', end=' ')
            torch.save(optim_state_dict, optim_save_path)
            dist.loginfo0('Done!')

        torch.distributed.barrier()

    if len(batch_sizes) > 1:
        best_batch_size_idx = int(np.argmax(bandwidths))
        dist.loginfo0(f'------------ Best batch size is {batch_sizes[best_batch_size_idx]} ------------')
    if full_summary:
        dist.loginfo0(summaries[best_batch_size_idx])

#----------------------------------------------------------------------------

if __name__ == "__main__":
    launch_from_registry(profile_model)

#----------------------------------------------------------------------------
