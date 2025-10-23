import os
import sys
import time
import math
import traceback
import functools
import copy
from contextlib import nullcontext
from datetime import datetime
from collections import defaultdict

import wandb
import hydra
import torch
from torch.distributed.elastic.multiprocessing.errors import record
from omegaconf import DictConfig

from src.structs import EasyDict, TensorGroup
from src.utils.os_utils import lean_tqdm, disable_trivial_warnings
from src.utils import distributed as dist, misc, logging_utils
from src.utils.autoencoder_utils import compute_autoencoder_stats
from src.metrics import metric_main
from src.inference import sample
from src.training import network_utils
from src.data import Data, construct_full_cond_from_batch, construct_inputs_from_batch, get_dp_data_subset
from src.structs import Nets, Optimizers, ResumeConfig, LossPhase
from src.data.utils import inf_loop_dataloader

#----------------------------------------------------------------------------

@record
@hydra.main(config_path="..", config_name="experiment_config.yaml", version_base=None)
def train(cfg: DictConfig):
    disable_trivial_warnings()
    cfg = EasyDict.init_recursively(cfg)
    dist.init(timeout_seconds=cfg.dist_timeout_seconds)
    dist.ControlSignal.clear_file_state()
    ctrl_state = dist.ControlSignal() # Initialize an empty control state.
    dist.loginfo0(f'Started training! Current time is: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    logging_utils.maybe_init_wandb(cfg)
    logging_utils.init_custom_logger(cfg.output_dir)

    resume_cfg = ResumeConfig.init_from_experiment_cfg(cfg)
    seed = init_random_seed(cfg)
    device = torch.device('cuda')

    _start_time = time.time()
    dist.init_random_state_and_cuda(seed, cfg.training.cudnn_benchmark, allow_tf32=cfg.training.allow_tf32)

    data: Data = Data.init_from_cfg(cfg.dataset)
    logging_utils.print_training_options(cfg, data, resumed_snapshot=resume_cfg.ckpt_path)

    # TODO: split the initializion into multiple simpler functions.
    nets, optimizers, loss_fn, vis, stats = network_utils.init_modules(cfg=cfg, device=device, start_time=_start_time, data=data, resume_cfg=resume_cfg)
    nets_to_log = nets if cfg.wandb.log_ema_stats else Nets(net=nets.net, ema=None)
    net_named_parameters = ema_named_parameters = None # dict cache for the model.

    # Note: we need to print the model info before computing the stats since it inits FSDP.
    logging_utils.print_model_info(cfg, data.batch_info.batch_gpu, nets, device)
    logging_utils.log_real_samples(cfg, nets, vis, stats, device)
    torch.distributed.barrier() # Wait for the main rank to finish the logging.

    if cfg.model.is_lgm and cfg.model.is_diffusion:
        init_ldm_autoencoder_stats(nets, data, cfg, device, stats)

    # Train.
    dist.loginfo0(f'Training for {cfg.training.max_steps} iterations...', end='\n')
    tick_stats = logging_utils.TickStats.init_from_training_stats(stats)
    maintenance_time = tick_stats.start_time - stats.start_time
    has_tick_just_started = True

    if cfg.profiling.enabled and (dist.is_main_process() or cfg.profiling.trace_all_ranks):
        profiler = misc.init_profiler(cfg.profiling.logs_dir)
        dist.loginfo0(f"Profiler is enabled. Logs will be saved to {cfg.profiling.logs_dir}")
        profiler.start()
    else:
        profiler = None

    if cfg.training.save_init and stats.cur_step == 0:
        snapshot_path = network_utils.get_snapshot_path(cfg, stats, snapshot_suffix='init')
        network_utils.save_snapshot(cfg, nets, optimizers, data, stats, vis, snapshot_path, reason='init', save_non_empty_optim=True)

    dist.loginfo0('Entering the main training loop...')

    while not stats.done:
        if not profiler is None:
            profiler.step()
        # Perfom logging before the update so we catch the init (which might be important) at the very beginning.
        pre_tick_maintenance_start_time = time.time()
        # This `should_log_param_stats` is a bit dirty. We use it in two places, and we also do an extra log right after the init (if param logging is enabled).
        should_log_param_stats = cfg.wandb.enabled and (cfg.training.freqs.param_stats is not None) and (stats.done or stats.cur_step % cfg.training.freqs.param_stats == 0)
        should_wandb_commit = cfg.wandb.enabled and (cfg.training.freqs.loss_train_commit is not None) and (stats.cur_step % cfg.training.freqs.loss_train_commit == 0 or stats.is_first_tick_after_launch)
        should_update_ema = stats.cur_step % cfg.model.ema_update_freq == 0

        if (cfg.training.freqs.save_samples is not None) and (stats.done or stats.cur_step % cfg.training.freqs.save_samples == 0) or ctrl_state.should_save_samples:
            log_generated_samples(cfg, nets.ema or nets.net, vis, data.dataset_val, stats, device)
        if (cfg.training.freqs.loss_per_sigma is not None) and (stats.done or stats.cur_step % cfg.training.freqs.loss_per_sigma == 0):
            log_per_sigma_losses(loss_fn, data.dataset_val, stats, cfg, nets.net, device, vis.sigma)
        if (cfg.training.freqs.loss_val is not None) and (stats.done or stats.cur_step % cfg.training.freqs.loss_val == 0):
            log_val_losses(loss_fn, data.dataset_val, stats, cfg, nets.net, device)

        if should_log_param_stats and stats.cur_step == 0:
            if cfg.model.is_autoencoder and cfg.model.log_latent_stats:
                compute_and_log_autoencoder_stats(nets_to_log, data, cfg, device, stats)
            with misc.summon_full_params(nets.net), (misc.summon_full_params(nets.ema) if nets_to_log.ema is not None else nullcontext()):
                if loss_fn.teacher is not None and cfg.training.log_teacher_params_diff:
                    logging_utils.log_params_diff(misc.unwrap_module(nets.net), loss_fn.teacher, stats)
                logging_utils.log_param_stats(nets_to_log._asdict(), stats)
        pre_tick_maintenance_time = time.time() - pre_tick_maintenance_start_time

        # Starting the tick
        if has_tick_just_started:
            tick_stats.start_time = time.time()
            tick_stats.start_nimg = stats.cur_nimg
            tick_stats.start_step = stats.cur_step
            has_tick_just_started = False

        should_do_D_step = False
        should_do_GD_step = should_do_D_step and cfg.discr.fuse_gd_step

        # Zero grad all the optimizers.
        optimizers.optimizer.zero_grad(set_to_none=True)

        # Accumulate gradients.
        with torch.profiler.record_function('compute_gradients'):
            phase = LossPhase.GenAndDiscr if should_do_GD_step else LossPhase.Gen
            if cfg.loss.get('dec_lipschitz_reg') and cfg.loss.dec_lipschitz_reg.r1_weight > 0 and stats.cur_step % cfg.loss.dec_lipschitz_reg.r1_freq == 0:
                with misc.disable_grad(misc.unwrap_module(nets.net).get_encoder_parameters()):
                    losses_r1_reg, _num_samples_processed = compute_gradients(cfg, nets.net, data, stats, loss_fn, optimizers.grad_scaler, device, phase=LossPhase.GenLipReg, prevent_ddp_sync=True)
            else:
                losses_r1_reg = None
            losses, num_samples_processed = compute_gradients(cfg, nets.net, data, stats, loss_fn, optimizers.grad_scaler, device, phase=phase)
            if losses_r1_reg is not None:
                for loss_key, loss_val in losses_r1_reg.items():
                    losses[loss_key] += loss_val
            loss_means = {loss_key: compute_mean_loss(losses_list, cfg) for loss_key, losses_list in losses.items()} # {num_domains, [batch_size]}
            del losses

        # Scale the gradients.
        with torch.profiler.record_function('grad_scaling'):
            if optimizers.grad_scaler is not None:
                optimizers.grad_scaler.unscale_(optimizers.optimizer)

        # Prepare the optimizer LRs.
        with torch.profiler.record_function('lr_scheduler_set_nimg'):
            if hasattr(optimizers.lr_scheduler, 'set_cur_nimg'):
                optimizers.lr_scheduler.set_cur_nimg(stats.cur_nimg)

        # Cleaning and clipping the gradients of the main network.
        with torch.profiler.record_function('clean_and_clip_grads'):
            net_grad_log_dict = maybe_clean_grad_(cfg.training.grad_nan_to_num, nets.net)
            grad_norm_log_dict = maybe_grad_clip_(cfg.model.grad_clip, nets.net)

        # Logging the parameter and latent statistics.
        if should_log_param_stats:
            misc.free_memory() # summon_full_params can be very expensive. We should be as lean as possible.
            if cfg.model.is_autoencoder and cfg.model.log_latent_stats:
                compute_and_log_autoencoder_stats(nets_to_log, data, cfg, device, stats)
            with torch.profiler.record_function('log_param_stats'), misc.summon_full_params(nets.net), (misc.summon_full_params(nets.ema) if nets_to_log.ema is not None else nullcontext()):
                if loss_fn.teacher is not None and cfg.training.log_teacher_params_diff:
                    logging_utils.log_params_diff(misc.unwrap_module(nets.net), loss_fn.teacher, stats)
                logging_utils.log_param_stats(nets_to_log._asdict(), stats)
                weights_copy_before_step = {n: {k: misc.full_tensor(p).cpu().detach().clone() for k, p in m.named_parameters()} for n, m in nets_to_log._asdict().items() if m is not None and not 'ema' in n}

        # Perform the optimization step for the main network.
        with torch.profiler.record_function('optim_step'):
            if optimizers.grad_scaler is None:
                optimizers.optimizer.step()
            else:
                optimizers.grad_scaler.step(optimizers.optimizer)
                optimizers.grad_scaler.update()

        # LR scheduler steps.
        with torch.profiler.record_function('lr_scheduler_step'):
            optimizers.lr_scheduler.step()

        if should_log_param_stats:
            with torch.profiler.record_function('log_param_update_stats'), misc.summon_full_params(nets.net):
                logging_utils.log_param_update_stats(nets_to_log._asdict(), weights_copy_before_step, stats)
                del weights_copy_before_step

        with torch.profiler.record_function('wandb_log'):
            logging_utils.maybe_log_losses_and_grads(cfg, stats, loss_means, net_grad_log_dict, grad_norm_log_dict, commit=should_wandb_commit)

        # Maybe update the EMA.
        ema_beta, net_named_parameters, ema_named_parameters = update_ema(cfg, nets.net, nets.ema, stats, net_named_parameters, ema_named_parameters) if should_update_ema and cfg.model.use_ema else (1.0, None, None)

        with torch.profiler.record_function('stats_log_and_update'):
            # Let's save some stats
            if cfg.wandb.enabled:
                net_progress_stats = {'Progress/lr': optimizers.optimizer.param_groups[0]['lr'], 'Progress/ema_beta': ema_beta}
                if cfg.model.is_vae:
                    net_progress_stats = {**net_progress_stats, **{'Progress/kl_loss_weight': loss_fn.get_cur_kl_loss_weight(stats.cur_step)}}
                if cfg.loss.name == 'alphaflow':
                    net_progress_stats = {**net_progress_stats, **{'Progress/alpha': loss_fn.get_ratio(cfg.loss.alpha, stats.cur_step)}}
                    net_progress_stats = {**net_progress_stats, **{'Progress/ratio_fm': loss_fn.get_ratio(cfg.loss.ratio_fm, stats.cur_step)}}
                logging_utils.wandb_log_with_stats0(net_progress_stats, stats, commit=should_wandb_commit)

            # Perform maintenance tasks once per tick.
            stats.cur_nimg += num_samples_processed
            stats.cur_step += 1
            stats.done = stats.cur_step >= cfg.training.max_steps

        with torch.profiler.record_function('ctrl_state_update'):
            should_check_ctrl = cfg.training.freqs.ctrl_check is not None and stats.cur_step % cfg.training.freqs.ctrl_check == 0
            ctrl_state = dist.ControlSignal.receive_from_file().sync() if should_check_ctrl else dist.ControlSignal()
            should_increment_tick_due_to_time = cfg.training.seconds_per_tick is not None and (time.time() - tick_stats.start_time) >= cfg.training.seconds_per_tick
            should_increment_tick_due_to_steps = cfg.training.steps_per_tick is not None and (stats.cur_step >= tick_stats.start_step + cfg.training.steps_per_tick)
            should_continue_tick = (not stats.done) and (stats.cur_tick != 0) and not should_increment_tick_due_to_time and not should_increment_tick_due_to_steps and not ctrl_state.should_do_something
            should_continue_tick = torch.tensor(should_continue_tick, dtype=torch.bool).to(device) # [1]
            if torch.distributed.is_initialized():
                torch.distributed.broadcast(should_continue_tick, src=0) # [1]
            should_continue_tick = should_continue_tick.item()

        if not should_continue_tick:
            # Freeing the previous memory. Can be expensive, so doing it once per tick only before doing the maintenance.
            misc.free_memory()

            # Print the status line.
            tick_stats.end_time = time.time()
            logging_utils.print_tick_stats0(stats, tick_stats, loss_means['total'].mean(), device, maintenance_time)
            if cfg.wandb.enabled:
                sec_per_nimg = (tick_stats.end_time - tick_stats.start_time) / (stats.cur_nimg - tick_stats.start_nimg + 1e-16)
                progress_stats_to_log = {
                    'Progress/sec_per_kimg': sec_per_nimg * 1000,
                    'Progress/sec_per_kimg_on_gpu': sec_per_nimg * 1000 * dist.get_world_size(),
                    'Progress/samples_per_sec': 1.0 / (sec_per_nimg + 1e-16),
                    'Progress/samples_per_sec_on_gpu': 1.0 / (sec_per_nimg + 1e-16) / dist.get_world_size(),
                    'Progress/iter_per_sec': (stats.cur_step - tick_stats.start_step) / (tick_stats.end_time - tick_stats.start_time)
                }
                logging_utils.wandb_log_with_stats0(progress_stats_to_log, stats)
            torch.cuda.reset_peak_memory_stats()

        # Check for abort.
        if stats.done or ctrl_state.should_stop:
            stats.done = True
            dist.print0()
            dist.loginfo0('Training is done! Aborting... ')

        evaluate_and_snapshot(cfg, stats, nets, optimizers, data, vis, device, should_ctrl_snapshot=ctrl_state.should_snapshot)

        if not should_continue_tick:
            # Update state for the new tick.
            stats.cur_tick += 1
            stats.is_first_tick_after_launch = False
            has_tick_just_started = True
            maintenance_time = time.time() - tick_stats.end_time + pre_tick_maintenance_time
            nets.net.train().requires_grad_(True) # Let's make sure that we are in the train mode before the next tick starts.
            if cfg.model.is_lgm and hasattr(nets.net, 'autoencoder') and isinstance(nets.net.autoencoder, torch.nn.Module):
                nets.net.autoencoder.eval().requires_grad_(False)

    # Done.
    if cfg.wandb.enabled and (dist.is_main_process() or cfg.wandb.distributed):
        wandb.finish()
    if not profiler is None:
        profiler.stop()
    dist.print0()
    dist.loginfo0('Exiting... ')

#----------------------------------------------------------------------------

def compute_metrics(cfg, metrics_to_compute, metrics_all, data: Data, net, snapshot_path, device, stats) -> float:
    if len(metrics_to_compute) > 0:
        dist.loginfo0(f'Evaluating [{", ".join(metrics_to_compute)}] metrics for {cfg.output_dir} ... ')

    curr_main_metric_value = float('inf')
    sampling_cfg_overwrites = {'': {}} | cfg.training.metrics_extra_sampling_cfg_overwrites

    for sampling_cfg_overwrite_name, sampling_cfg_overwrite_values in sampling_cfg_overwrites.items():
        curr_sampling_cfg = copy.deepcopy(cfg.sampling)
        curr_sampling_cfg.update(sampling_cfg_overwrite_values)
        if sampling_cfg_overwrite_name == '':
            sampling_suffix = "consistency" if curr_sampling_cfg.enable_consistency_sampling else "ode"
            metric_name_suffix = f"NFE_{curr_sampling_cfg.num_steps}_{sampling_suffix}_sampling"
        else:
            metric_name_suffix = sampling_cfg_overwrite_name

        for metric in metrics_to_compute:
            result_dict = metric_main.compute_metric(
                metric=metric,
                net=net,
                batch_gen=cfg.dataset.gen_batch_gpu,
                detector_batch_gpu=cfg.dataset.metrics_detector_batch_gpu,
                verbose=True,
                sampling_cfg=curr_sampling_cfg,
                rank=dist.get_rank(),
                device=device,
                dataset_stats_dir=cfg.env.dataset_stats_dir,
                dataset=data.dataset_eval,
                metric_name_suffix=metric_name_suffix,
            )

            if cfg.training.main_metric == '__pick_first__' and metric == metrics_all[0]:
                curr_main_metric_value = result_dict['results'][next(iter(result_dict['results']))]
            if dist.is_main_process():
                metric_main.report_metric(result_dict, run_dir=cfg.output_dir, snapshot_path=snapshot_path)
            if dist.is_main_process() and cfg.wandb.enabled:
                logging_utils.wandb_log_with_stats0({**{f'Metrics/{k}': v for k, v in result_dict['results'].items()}}, stats)
            misc.free_memory()

    return curr_main_metric_value

#----------------------------------------------------------------------------

@torch.no_grad()
def log_generated_samples(cfg, net, vis, dataset_val, stats, device, save_name: str=None):
    sampling_cfg_overwrites = {'': {}} | cfg.training.metrics_extra_sampling_cfg_overwrites

    for sampling_cfg_overwrite_name, sampling_cfg_overwrite_values in sampling_cfg_overwrites.items():
        curr_sampling_cfg = copy.deepcopy(cfg.sampling)
        curr_sampling_cfg.update(sampling_cfg_overwrite_values)
        if sampling_cfg_overwrite_name == '':
            sampling_suffix = "consistency" if curr_sampling_cfg.enable_consistency_sampling else "ode"
            sampling_name_suffix = f"_NFE_{curr_sampling_cfg.num_steps}_{sampling_suffix}_sampling"
        else:
            sampling_name_suffix = "_" + sampling_cfg_overwrite_name

        dist.loginfo0(f'Logging generated samples for {sampling_name_suffix}... ', end='')
        # We can use a smaller gen_batch_gpu if we have enough ranks, but ensure at least 1.
        gen_batch_gpu = min(cfg.dataset.gen_batch_gpu, math.ceil(len(vis.grid_seeds) / dist.get_world_size()))
        # Temporary hack to avoid loading the heavy videos.
        grid_batch = [dataset_val[s.item() % len(dataset_val)] for s in lean_tqdm(vis.grid_seeds, desc='Loading sample conditioning for generation.', disable=not dist.is_main_process())] # (num_samples, [<sample>])
        grid_batch = torch.utils.data.default_collate(grid_batch) # [num_samples, <sample>]

        grid_seeds = vis.grid_seeds.split(gen_batch_gpu) # (num_batches, [batch_size])
        grid_cond = construct_full_cond_from_batch(grid_batch, cfg.model.use_x_cond).split(gen_batch_gpu) # (num_batches, [batch_size, ...])

        # Compute the vis group size which we would like to aggregate over in the future.
        # All the GPUs would be doing some sampling (we can't let them wait on a barrier due to FSDP),
        # but we'll aggregate the results only from the a subset of ranks.
        vis_group_size = min(dist.get_world_size(), len(grid_seeds))
        rank_seeds = grid_seeds[dist.get_rank()::vis_group_size] # (num_batches_on_gpu, [batch_size])
        rank_grid_cond = grid_cond[dist.get_rank()::vis_group_size] # (num_batches_on_gpu, [batch_size, *label_shape])
        rank_0_seeds = grid_seeds[0::vis_group_size]
        if len(rank_seeds) < len(rank_0_seeds):
            assert len(rank_seeds) + 1 == len(rank_0_seeds), f"The difference cannot be more that 1 batch: {len(rank_seeds), rank_0_seeds}"
            rank_seeds = rank_seeds + grid_seeds[-1:]
            rank_grid_cond = rank_grid_cond + grid_cond[-1:]
        # Generating the samples on all the ranks (might be crucial for FSDP).
        samples = sample(net, curr_sampling_cfg, rank_seeds, cond=rank_grid_cond, verbose=False, device=device, save_traj_len=vis.save_traj_len) # [num_samples, traj_len, t, c, h, w]
        samples = TensorGroup(video=samples) if not isinstance(samples, TensorGroup) else samples # [num_samples, traj_len, t, c, h, w], [num_samples, ca, ta]

        # Now, we are ready to aggregate the results from the vis group (instead of all the ranks).
        vis_group = torch.distributed.new_group(ranks=range(vis_group_size))
        if dist.get_rank() >= len(grid_seeds):
            torch.distributed.barrier() # Barrier all the processes except for those who participate in samples aggregation.
        else:
            samples = dist.gather_seeded_results(samples.to(device), torch.cat(rank_seeds).to(device), vis_group, cpu_offload_dims=[1,2,3]) # <k, [num_samples, traj_len, t, c, h, w]>
            samples = (TensorGroup(samples) if isinstance(samples, dict) else samples).cpu()
            torch.distributed.barrier()

        # We do not need this group anymore and can destroy it. Otherwise, there would be threads leaking.
        torch.distributed.destroy_process_group(group=vis_group)

        logging_utils.save_samples(samples, stats, cfg.output_dir, save_name=save_name, save_to_wandb=cfg.wandb.enabled, as_grid=vis.save_videos_as_grid, sampling_name_suffix=sampling_name_suffix)

        torch.distributed.barrier()
        del samples; misc.free_memory()
    dist.loginfo0('done!')

#----------------------------------------------------------------------------
# Validation losses computation and logging.

@torch.no_grad()
def log_val_losses(loss_fn, dataset_val, stats, cfg, net, device):
    if not cfg.wandb.enabled or cfg.training.num_samples_for_val_loss == 0:
        return
    dist.loginfo0('Logging validation losses... ', end='')
    loss_dict = compute_val_losses(loss_fn, dataset_val, stats, cfg, net, device, cfg.training.num_samples_for_val_loss)
    logging_utils.wandb_log_with_stats0(loss_dict, stats)
    del loss_dict; misc.free_memory()
    dist.loginfo0('done!')

@torch.no_grad()
def log_per_sigma_losses(loss_fn, dataset_val, stats, cfg, net, device, sigmas: torch.Tensor):
    assert sigmas.ndim == 1, f"Sigmas should be a 1D tensor, but got {sigmas.shape}."
    if not cfg.wandb.enabled: return

    losses_per_sigma = {}
    for sigma_idx, cur_sigma in lean_tqdm(enumerate(sigmas), desc='Logging per-sigma losses... ', disable=not dist.is_main_process(), total=len(sigmas)):
        loss_fn_cur = functools.partial(loss_fn, force_sigma_val=cur_sigma.item())
        loss_dict_cur = compute_val_losses(loss_fn_cur, dataset_val, stats, cfg, net, device, cfg.training.num_samples_for_per_sigma_loss)
        losses_per_sigma[f'Loss_per_sigma/level_idx_{sigma_idx:2d}'] = loss_dict_cur['Loss/val_rec'].mean().item()
        losses_per_sigma[f'Loss_per_sigma_unweighted/level_idx_{sigma_idx:2d}'] = loss_dict_cur['Loss/val_rec_unweighted'].mean().item()

    logging_utils.wandb_log_with_stats0(losses_per_sigma, stats)
    del losses_per_sigma; misc.free_memory()

@torch.no_grad()
def compute_val_losses(loss_fn, dataset_val, stats, cfg, net, device, num_samples_to_use: int):
    misc.free_memory()
    dataset_val.epoch = 0 # Resetting the epoch counter to start from the beginning.
    dataset_val_cur = get_dp_data_subset(dataset_val) if isinstance(dataset_val, torch.utils.data.Dataset) else dataset_val
    dataloader_val = torch.utils.data.DataLoader(dataset_val_cur, drop_last=True, batch_size=cfg.dataset.test_batch_gpu, num_workers=0, pin_memory=True)
    dataset_val_iterator = iter(inf_loop_dataloader(dataloader_val))
    loss_val_values = defaultdict(list)
    num_batches_to_process = max(num_samples_to_use // (cfg.dataset.test_batch_gpu * dist.get_world_size()), 1)
    assert num_batches_to_process > 0, f"Number of batches to process {num_batches_to_process} is not positive. Please increase the number of samples to process."
    for _ in range(num_batches_to_process):
        batch = next(dataset_val_iterator)
        x, cond = construct_inputs_from_batch(batch, use_x_cond=cfg.model.use_x_cond, device=device)
        losses_val = loss_fn(net=net, x=x, cond=cond, cur_step=stats.cur_step, phase=LossPhase.Gen) # [num_val_samples, num_pyr_levels]
        for loss_key, loss_val in losses_val.items():
            loss_val_values[loss_key].append(loss_val.mean(dim=0).cpu())
    loss_dict = {}
    for loss_key, losses in loss_val_values.items():
        loss_val_agg = torch.stack(losses).to(device).mean(dim=0) # [batch_size]
        loss_val_agg = dist.mean_across_gpus(loss_val_agg).cpu() # [batch_size]
        loss_dict = {**loss_dict, **{f'Loss/val_{loss_key}': loss_val_agg.mean().cpu()}}
    torch.distributed.barrier() # Setting a barrier for a longer timeout to make sure that all the samples are processed.
    del dataset_val, dataset_val_cur, dataloader_val, dataset_val_iterator, loss_val_values, loss_val_agg; misc.free_memory()
    return loss_dict

#----------------------------------------------------------------------------

@torch.no_grad()
def compute_mean_loss(losses_list: torch.Tensor, cfg) -> torch.Tensor:
    loss_means = torch.cat(losses_list, dim=0).mean(dim=0) # [b]
    loss_means = (dist.mean_across_gpus(loss_means) if cfg.training.log_loss_across_gpus else loss_means) # [b]
    return loss_means

#----------------------------------------------------------------------------
# Snapshotting utils.

def get_reason_to_snapshot(cfg, stats, curr_main_metric_value, best_metric_value, should_ctrl_snapshot: bool=False) -> str | None:
    """Calculates whether we should snapshot and what is the reason."""
    if stats.done:
        return 'done'
    elif curr_main_metric_value <= best_metric_value and curr_main_metric_value != float('inf'):
        return 'best_metric'
    elif not cfg.training.freqs.snapshot is None and stats.cur_step % cfg.training.freqs.snapshot == 0:
        return 'step'
    elif not cfg.training.freqs.snapshot is None and stats.cur_step == 1:
        return 'step_first'
    elif not cfg.training.freqs.snapshot_latest is None and stats.cur_step % cfg.training.freqs.snapshot_latest == 0:
        return 'step_latest'
    elif should_ctrl_snapshot:
        return 'ctrl'
    else:
        return None

#----------------------------------------------------------------------------

@misc.profiled_function
@torch.no_grad()
def update_ema(cfg: EasyDict, net, ema, stats, net_named_parameters: dict=None, ema_named_parameters: dict=None) -> tuple[float, dict, dict]:
    cur_ema_halflife_steps = cfg.model.ema_halflife_steps
    if cfg.model.ema_rampup_ratio is not None:
        cur_ema_halflife_steps = min(cur_ema_halflife_steps, stats.cur_step * cfg.model.ema_rampup_ratio)
    ema_beta = 0.5 ** (1 / max(cur_ema_halflife_steps, 1e-8))
    # We use pre-computed named parameters to avoid the overhead of the named_parameters() call (it is sometimes quite expensive).
    net_named_parameters = dict(**dict(misc.unwrap_module(net).named_parameters()), **dict(misc.unwrap_module(net).named_buffers())) if net_named_parameters is None else net_named_parameters
    ema_named_parameters = dict(**dict(misc.unwrap_module(ema).named_parameters()), **dict(misc.unwrap_module(ema).named_buffers())) if ema_named_parameters is None else ema_named_parameters
    for param_name in net_named_parameters:
        p_net_param = net_named_parameters[param_name]
        if not p_net_param.dtype.is_floating_point:
            continue
        p_ema_param = ema_named_parameters[param_name]
        p_ema_param.lerp_(p_net_param, 1.0 - ema_beta) # A faster in-place operation.
    return float(ema_beta), net_named_parameters, ema_named_parameters

#----------------------------------------------------------------------------

def compute_gradients(cfg, net_dp, data: Data, stats, loss_fn, grad_scaler, device, phase: LossPhase, prevent_ddp_sync: bool=False) -> tuple[dict[str, torch.Tensor], int]:
    losses_log_dict: dict[str, list] = defaultdict(list)
    num_accum_rounds_left = num_accum_rounds = data.batch_info.num_accum_rounds
    num_samples_processed = 0
    stream_names = set()

    while True: # We'll decide along the way when to stop for the current iteration.
        with torch.profiler.record_function(f'data_fetch'):
            batch = next(data.iterator)
            num_samples_processed += batch['video'].shape[0] * dist.get_world_size() # [b, t, h, w, c]

        stream_names.add(getattr(batch, 'stream_name', None))
        assert len(stream_names) == 1, f"Expected only one stream type in the batch, but got: {stream_names}."
        if num_accum_rounds is None:
            num_accum_rounds = getattr(batch, 'num_accum_rounds_left', 0) + 1
            num_accum_rounds_left = num_accum_rounds
        num_accum_rounds_left -= 1

        with torch.profiler.record_function('data_to_device'):
            x, cond = construct_inputs_from_batch(batch, use_x_cond=cfg.model.use_x_cond, device=device) # [b, ...], [b, ...]

        with torch.profiler.record_function(f'loss_fn_{phase}'), misc.maybe_ddp_sync(net_dp, force_sync=num_accum_rounds_left==0 and not prevent_ddp_sync):
            losses = loss_fn(net=net_dp, x=x, cond=cond, phase=phase, cur_step=stats.cur_step) # <str, [batch_size, num_levels, t, c, h, w] | [batch_size, num_levels]>

        # Now, we are safe to process the G loss.
        if phase.has_gen_update():
            loss_G_total_cur = losses.total.mean() / num_accum_rounds # [1]
            loss_G_total_cur = cfg.model.loss_scaling * (loss_G_total_cur if grad_scaler is None else grad_scaler.scale(loss_G_total_cur)) # [1]
            loss_G_total_cur.backward()

        # Save the losses for logging.
        for loss_key, loss in losses.items():
            losses_log_dict[loss_key].append(loss.detach())

        if num_accum_rounds_left == 0:
            break

    return losses_log_dict, num_samples_processed

#----------------------------------------------------------------------------

def evaluate_and_snapshot(cfg, stats, nets: Nets, optimizers: Optimizers, data: Data, vis, device, should_ctrl_snapshot: bool=False):
    # TODO: move from `metrics_rare` to just specifying metric log frequency.
    snapshot_path = network_utils.get_snapshot_path(cfg, stats)
    curr_metrics = cfg.training.metrics if not cfg.training.freqs.metrics is None and (stats.done or stats.cur_step % cfg.training.freqs.metrics == 0 or stats.cur_step == 1) else []
    curr_metrics_rare = cfg.training.metrics_rare if not cfg.training.freqs.metrics_rare is None and (stats.done or (stats.cur_step % cfg.training.freqs.metrics_rare == 0 and not stats.is_first_tick_after_launch)) else []
    curr_metrics_final = cfg.training.metrics_final if stats.done else []
    curr_main_metric_value = compute_metrics(cfg, curr_metrics + curr_metrics_rare + curr_metrics_final, cfg.training.metrics, data, nets.ema or nets.net, snapshot_path, device, stats) # [1]
    reason_to_snapshot = get_reason_to_snapshot(cfg, stats, curr_main_metric_value, stats.best_metric_value, should_ctrl_snapshot=should_ctrl_snapshot)

    if reason_to_snapshot is not None:
        try:
            stats.snapshot_descs[snapshot_path] = reason_to_snapshot
            network_utils.save_snapshot(cfg, nets, optimizers, data, stats, vis, snapshot_path, reason=reason_to_snapshot)
            if dist.is_main_process() and reason_to_snapshot == 'step_latest':
                # Delete the previous latest snapshot (if any)
                snapshots_to_delete = [p for p, reason in stats.snapshot_descs.items() if reason == 'step_latest' and p != snapshot_path]
                for p in snapshots_to_delete:
                    if os.path.exists(p):
                        os.remove(p)
                    del stats.snapshot_descs[p]
        except Exception as e: # pylint: disable=broad-except
            dist.loginfo0(f'Failed to save the checkpoint with error: {str(e)}')
            traceback.print_exc()

    # Update the best metric value
    # TODO: we can do this via wandb, right?
    if curr_main_metric_value <= stats.best_metric_value and curr_main_metric_value != float('inf'):
        if dist.is_main_process():
            prev_best_ckpt_path = os.path.join(cfg.output_dir, f'snapshot-{stats.best_metric_nimg//1000:06d}.pt')
            if (not cfg.training.freqs.snapshot is None and stats.best_metric_step % cfg.training.freqs.snapshot == 0) or stats.done:
                pass # Do not delete the snapshot since we would save it anyway
            elif os.path.isfile(prev_best_ckpt_path):
                # Deleting the previous best
                dist.loginfo0(f'Removing the previous best ckpt: {prev_best_ckpt_path}... ', end='')
                os.remove(prev_best_ckpt_path)
                dist.loginfo0('Done!')
        # Updating the best ckpt statistics
        stats.best_metric_step = stats.cur_step
        stats.best_metric_nimg = stats.cur_nimg
        stats.best_metric_value = curr_main_metric_value

#----------------------------------------------------------------------------

def init_ldm_autoencoder_stats(nets: Nets, data, cfg, device, stats: logging_utils.TrainingStats):
    if misc.unwrap_module(nets.net).are_ldm_stats_inited.bool():
        assert nets.ema is None or misc.unwrap_module(nets.ema).are_ldm_stats_inited.bool(), "The stats are inited for the net, but not for the ema."
        return

    if cfg.model.latents_mean is not None and cfg.model.latents_std is not None:
        dist.loginfo0(f'Found latents mean and std in the config: {cfg.model.latents_mean}, {cfg.model.latents_std}. Skipping the stats estimation.')
        latents_mean = torch.as_tensor(cfg.model.latents_mean).expand_as(misc.unwrap_module(nets.net).autoencoder_mean) # [c]
        latents_std = torch.as_tensor(cfg.model.latents_std).expand_as(misc.unwrap_module(nets.net).autoencoder_var) # [c]
        num_samples_used = 0
    else:
        stats_dict = compute_autoencoder_stats(misc.unwrap_module(nets.net).autoencoder, data, cfg, device, cfg.model.num_samples_to_estimate_stats, stats_keys=['latents'])
        logging_utils.log_autoencoder_stats(stats_dict, save_to_wandb=cfg.wandb.enabled, training_stats=stats, wandb_prefix='LDM_autoencoder_stats')
        num_samples_used = stats_dict.latents.num_samples_used
        latents_mean, latents_std = torch.from_numpy(stats_dict.latents.mean), torch.from_numpy(stats_dict.latents.std)
        latents_mean = latents_mean if cfg.model.latents_mean is None else torch.as_tensor(cfg.model.latents_mean).expand_as(latents_mean) # [lt, lh, lw, ?]
        latents_std = latents_std if cfg.model.latents_std is None else torch.as_tensor(cfg.model.latents_std).expand_as(latents_std) # [lt, lh, lw, ?]

    for model in [nets.net, nets.ema]:
        if model is None: continue
        misc.unwrap_module(model).set_autoencoder_stats(latents_mean, latents_std.pow(2), num_samples_used)


def compute_and_log_autoencoder_stats(nets: Nets, data: Data, cfg, device, stats: logging_utils.TrainingStats):
    for model_type, autoencoder in [('net', nets.net), ('ema', nets.ema)]:
        if autoencoder is None: continue
        stats_dict = compute_autoencoder_stats(autoencoder, data, cfg, device, cfg.model.num_samples_to_estimate_stats_for_logging)
        logging_utils.log_autoencoder_stats(stats_dict, save_to_wandb=cfg.wandb.enabled, training_stats=stats, wandb_prefix=f'{model_type}_autoencoder_stats')

#----------------------------------------------------------------------------

def init_random_seed(cfg: EasyDict) -> int:
    if cfg.training.seed is not None:
        return cfg.training.seed
    else:
        seed = torch.randint(1 << 31, size=[], device=torch.device('cuda'))
        torch.distributed.broadcast(seed, src=0)
        return int(seed)

#----------------------------------------------------------------------------
# Gradient cleaning/cliping utilities.

def maybe_grad_clip_(grad_clip_cfg: EasyDict, net: torch.nn.Module, suffix: str="") -> dict:
    with torch.profiler.record_function(f'grad_clip{suffix}'):
        # TODO: apparently, gradient clipping does go well with FSDP and clips only on a shard.
        # I guess we can have it fixed with summon_full_params, but currently it works better without it.
        # Apparently, the difference is in the magnitude. We should ablate this.
        if not grad_clip_cfg.norm is None:
            if isinstance(net, torch.distributed.fsdp.FullyShardedDataParallel) and grad_clip_cfg.fsdp_clip_all_params:
                norm = net.clip_grad_norm_(grad_clip_cfg.norm) # [1]
            else:
                norm = torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip_cfg.norm) # [1]
            grad_norm_log_dict = {f'Grad/norm{suffix}': norm}
        else:
            grad_norm_log_dict = {}
        if not grad_clip_cfg.value is None:
            torch.nn.utils.clip_grad_value_(net.parameters(), grad_clip_cfg.value)
    return grad_norm_log_dict

def maybe_clean_grad_(grad_nan_to_num: bool, net: torch.nn.Module, suffix: str="") -> dict:
    with torch.profiler.record_function(f'grad_nan_to_num{suffix}'):
        if grad_nan_to_num:
            net_grad_stats = misc.grad_nan_to_num_(net.named_parameters(), return_stats=True)
            # net_grad_log_dict = {f'Grad_stats/net_{stat_type}/{param_name}': stat_val for param_name in net_grad_stats for stat_type, stat_val in net_grad_stats[param_name].items()}
            net_grad_log_dict = {f'Grad/net{suffix}_{stat_type}': stat_val for stat_type, stat_val in net_grad_stats.total.items()}
        else:
            net_grad_log_dict = {}
    return net_grad_log_dict

#----------------------------------------------------------------------------

if __name__ == "__main__":
    train() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
