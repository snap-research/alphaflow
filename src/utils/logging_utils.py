import os
import time
import math
import subprocess
import tempfile
import traceback
import os
import sys
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Any, Callable, Optional, Union

import psutil
import numpy as np
from PIL import Image
import torch
import torchvision
import einops
import wandb
try:
    import torchaudio
except ImportError:
    torchaudio = None


from src.structs import EasyDict, TensorGroup
from src.utils import misc, distributed as dist
from src.utils.os_utils import lean_tqdm, format_time
from src.structs import BasicStats, BasicStatsAgg, Nets, BaseLoss
from src.data import Data
from src.inference import sample_batch_diffusion

#----------------------------------------------------------------------------

@dataclass(frozen=False)
class TrainingStats:
    cur_nimg: int = 0
    cur_tick: int = 0
    cur_step: int = 0
    best_metric_value: float = float('inf')
    best_metric_step: int = 0 # Tracking at which tick the best metric was encountered
    best_metric_nimg: int = 0 # Tracking at which nimg the best metric was encountered
    snapshot_descs: EasyDict = None # Tracking what was the snapshotting reason is (e.g., best metric, training is done, timer, etc.).
    is_first_tick_after_launch: bool = True
    done: bool = False
    start_time: float = datetime.now().timestamp()

    @staticmethod
    def init_from_snapshot(snapshot: Optional[EasyDict], start_time: float=None, resume_whole_state: bool=False, allow_missing_state: bool=False) -> "TrainingStats":
        if resume_whole_state and not allow_missing_state and (snapshot is None or 'stats' not in snapshot):
            raise ValueError("Could not find stats in the snapshot while required to resume the whole state.")

        if snapshot is not None and 'stats' in snapshot:
            state = snapshot['stats']
            return TrainingStats(
                cur_nimg                   = state['cur_nimg'],
                cur_tick                   = state['cur_tick'],
                cur_step                   = state['cur_step'],
                best_metric_value          = state['best_metric_value'],
                best_metric_step           = state['best_metric_step'],
                best_metric_nimg           = state['best_metric_nimg'],
                snapshot_descs             = state['snapshot_descs'],
                start_time                 = start_time,
            )
        else:
            return TrainingStats(start_time=start_time, snapshot_descs={})

#----------------------------------------------------------------------------

@dataclass(frozen=False)
class TickStats:
    start_nimg: int
    start_step: int
    start_time: float
    end_time: float

    @staticmethod
    def init_from_training_stats(stats: TrainingStats) -> "TickStats":
        return TickStats(start_nimg=stats.cur_nimg, start_step=stats.cur_step, start_time=time.time(), end_time=None)

#----------------------------------------------------------------------------

def load_vis_data(cfg: EasyDict, loss: BaseLoss, dataset) -> EasyDict:
    vis = EasyDict()
    num_samples = cfg.training.num_vis_samples or 8
    vis.num_log_sigmas = cfg.training.num_log_sigmas
    vis.save_videos_as_grid = (max(cfg.dataset.resolution) <= 512 if cfg.dataset.resolution[0] == 1 else max(cfg.dataset.resolution) <= 256)
    vis.save_traj_len = cfg.training.traj_len_for_vis_gen or min(4 if max(cfg.dataset.resolution) >= 512 else 10, cfg.sampling.num_steps + 1)
    num_real_samples_for_vis = cfg.training.num_vis_samples or (16 if max(cfg.dataset.resolution) >= 512 else (25 if max(cfg.dataset.resolution) >= 256 else (36 if max(cfg.dataset.resolution) >= 128 else 64)))
    sample_ids = np.arange(num_real_samples_for_vis) # [num_samples_for_vis]
    if hasattr(dataset, 'random_resize'):
        # TODO: that's a very dirty hack.
        random_resize = dataset.random_resize
        dataset.random_resize = None
    samples = torch.utils.data.dataloader.default_collate([dataset[i] for i in lean_tqdm(sample_ids, desc='Loading GT samples for visualization', disable=not dist.is_main_process())]) # This would reuqire some downloading...
    if hasattr(dataset, 'random_resize'):
        dataset.random_resize = random_resize
    vis.videos = samples['video'] if dist.is_main_process() else None # [num_samples, t, c, h, w] or None
    vis.audio = samples['audio'] if dist.is_main_process() and 'audio' in samples else None
    vis.grid_seeds = torch.arange(num_samples) # [batch_size]
    vis.captions = samples['text'] if cfg.dataset.has_captions else []
    all_sigmas = loss.sample_sigma([1_000_000, 1, 1, 1, 1], device='cpu').squeeze() # [1_000_000, 1, 1, 1, 1]
    if isinstance(all_sigmas, TensorGroup):
        vis.sigma = TensorGroup({k: torch.linspace(v.min(), v.max(), vis.num_log_sigmas) for k, v in all_sigmas.items()}) # [num_log_sigmas]
    else:
        vis.sigma = torch.linspace(all_sigmas.min(), all_sigmas.max(), vis.num_log_sigmas) # [num_log_sigmas]
    return vis

#----------------------------------------------------------------------------

def save_samples(samples: TensorGroup, stats, output_dir: os.PathLike, save_name: str=None, save_to_wandb: bool=False, as_grid: bool=False, seeds: list[int] | None=None, use_res_suffix: bool=True, captions: list[str] | None=None, sampling_name_suffix: str='', **saving_kwargs):
    assert isinstance(samples, TensorGroup), f'samples must be a TensorGroup, got {type(samples)}'
    waveforms = samples.audio if 'audio' in samples else None # [num_samples, ca, ta] or None
    videos = samples.video if 'video' in samples else None # [num_samples, traj_len | None, t, h, w, c] or None

    # Saves videos generated from multiple stages. Shape is assumed to be [num_samples, traj_len, t, h, w, c]
    assert videos is None or videos.ndim in (5, 6) and videos.shape[-1] in (1, 3, 4), f"Wrong video shape: {videos.shape}"
    assert waveforms is None or waveforms.ndim in (3, 4), f"Wrong shape of audio: {waveforms.shape if waveforms is not None else None}"

    videos = videos if videos is None or videos.ndim == 6 else videos.unsqueeze(1) # [num_samples, traj_len | 1, t, h, w, c]
    res_suffix = f'_{videos.shape[2]}x{videos.shape[3]}x{videos.shape[4]}' if use_res_suffix else ''
    file_save_name = f'fakes_{stats.cur_step:08d}' if save_name is None else save_name
    wandb_save_name = f"{'fakes' if save_name is None else save_name}{res_suffix}{sampling_name_suffix}"
    file_save_path = os.path.join(output_dir, f'{file_save_name}{res_suffix}{sampling_name_suffix}')

    if as_grid:
        assert waveforms is None, "Audio logging is not supported when saving videos as grid."
        num_samples_per_row = videos.shape[1] if videos.shape[1] > 1 else math.floor(np.sqrt(len(videos)))
        videos = einops.rearrange(videos, 'b traj_len t h w c -> t (b traj_len) c h w') # [t, num_samples * traj_len, t, c, h, w]
        grid = torch.stack([torchvision.utils.make_grid(v, nrow=num_samples_per_row) for v in videos]).permute(0, 2, 3, 1) # [t, gh, gw, c]
        save_video(grid.numpy(), file_save_path, wandb_log=save_to_wandb, stats=stats, wandb_name=wandb_save_name, **saving_kwargs)
    else:
        videos = einops.rearrange(videos, 'b traj_len t h w c -> b t h (traj_len w) c') # [num_samples, t, h, traj_len * w, c]
        seeds = list(range(len(videos))) if seeds is None else list(seeds)
        captions = ([None] * len(videos)) if captions is None else list(captions)
        assert len(seeds) == videos.shape[0], f"Wrong number of seeds: {len(seeds) if seeds is not None else None}, expected: {videos.shape[0]}"
        for sample_idx, (seed, caption, v) in enumerate(zip(seeds, captions, videos)):
            waveform = waveforms[sample_idx].numpy() if waveforms is not None else None
            waveform = waveform[-1] if waveform is not None and waveform.ndim == 3 else waveform # [ca, ta] or None
            save_video(v.numpy(), f'{file_save_path}_{seed:06d}', wandb_log=save_to_wandb, stats=stats, wandb_name=f'{wandb_save_name}_{seed:06d}', waveform=waveform, caption=caption, **saving_kwargs)

#----------------------------------------------------------------------------

def save_video(video_np: np.ndarray, save_path: str, wandb_log: bool=False, stats: TrainingStats=None, wandb_name: str=None, caption: str=None, waveform=None, audio_sample_rate: int=16000, **kwargs):
    """
    Assumes videos to be of shape [l h w c], and the range is [0, 255], and dtype is uint8.
    """
    assert video_np.dtype == np.uint8, f"Wrong dtype: {video_np.dtype}"
    assert video_np.ndim == 4, f"Wrong shape: {video_np.shape}"
    assert video_np.shape[3] in (1, 3), f"Wrong shape: {video_np.shape}"
    ext = '.png' if video_np.shape[0] == 1 else '.mp4'
    save_path = f'{save_path}{ext}'

    if os.path.dirname(save_path) != '':
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if video_np.shape[0] == 1:
        video_np = video_np.squeeze(0)
        img = Image.fromarray(video_np, 'L') if video_np.shape[1] == 1 else Image.fromarray(video_np, 'RGB')
        img.save(save_path)
    else:
        retry_function(lambda: write_numpy_video_to_disk(video_np, save_path, waveform=waveform, audio_sample_rate=audio_sample_rate, **kwargs), num_retries=3, error_msg=f'Could not save {save_path} to disk')

    if caption is not None:
        with open(save_path[:-4] + '-caption.txt', 'w') as f:
            f.write(caption)

    if waveform is not None:
        assert torchaudio is not None, "torchaudio is not installed"
        torchaudio.save(save_path[:-4] + '.wav', torch.from_numpy(waveform), sample_rate=16000)

    # Uploading to wandb.
    if wandb_log:
        assert stats is not None, "stats must be provided if wandb_log is True"
        wandb_name = wandb_name if not wandb_name is None else os.path.basename(save_path)[:-4]
        wandb_cls = wandb.Image if video_np.ndim == 3 else wandb.Video
        retry_function(lambda: wandb_log_with_stats0({f"Samples/{wandb_name}": wandb_cls(save_path)}, stats), num_retries=3, error_msg=f'Could not save {save_path} to wandb')

#----------------------------------------------------------------------------

def write_numpy_video_to_disk(
    frames: np.ndarray,
    save_path: os.PathLike,
    fps: float = 25.0,
    threads: int = 1,
    crf: int = 22,
    pix_fmt: str = 'rgb24',
    save_gif: bool = False,
    waveform: np.ndarray = None, # [C, T], float32
    audio_codec='libmp3lame', # 'aac' is not readable in VSCode...
    audio_sample_rate: int = 16000,
):

    frames = np.concatenate([frames, 0 * frames[:, :1, :, :]], axis=1) if frames.shape[1] % 2 != 0 else frames
    frames = np.concatenate([frames, 0 * frames[:, :, :1, :]], axis=2) if frames.shape[2] % 2 != 0 else frames

    # Write temporary audio file if waveform is given
    if waveform is not None and waveform.shape[1] > 0:
        waveform = torch.from_numpy(waveform)
        assert waveform.ndim == 2, f"Expected [C, T], got {waveform.shape}"
        tmp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_audio_path = tmp_audio.name
        tmp_audio.close()
        torchaudio.save(tmp_audio_path, waveform, sample_rate=int(audio_sample_rate / (25 / fps)))
    else:
        tmp_audio_path = None

    command = [
        'ffmpeg',
        '-loglevel', 'error', # Reduce verbosity
        '-threads', str(threads),
        '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{frames.shape[2]}x{frames.shape[1]}',
        '-pix_fmt', pix_fmt,
        '-r', str(fps),
        '-i', '-',
    ]

    if tmp_audio_path:
        command += ['-i', tmp_audio_path]
        command += ['-strict', '-1'] # To save with audio sample rate of less than 8K hz.

    command += [
        '-vf', f'setpts={25/fps}*PTS', # FFMPEG assumes 25 FPS input framerate. We want to resample it to the given FPS.
        '-crf', str(crf),
        '-vcodec', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-x264opts', f'threads={threads}',
    ]

    if tmp_audio_path:
        command += ['-c:a', audio_codec, '-shortest']  # ensure video stops when shortest stream ends
    else:
        command += ['-an'] # no audio

    command += [str(save_path)]

    env_vars = dict(os.environ)
    env_vars['OMP_NUM_THREADS'] = '1'

    pipe = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, env=env_vars)
    pipe.communicate(input=frames.tobytes()) # Use communicate as a better alternative than stdin.write
    pipe.stdin.close()

    if save_gif:
        gif_cmd = ['ffmpeg', '-i', save_path, '-vf', f'fps={fps}', '-loglevel', 'error', '-y', '-loop', '0', save_path[:-4] + '.gif']
        subprocess.run(gif_cmd, check=True)

    if tmp_audio_path:
        os.remove(tmp_audio_path)

#----------------------------------------------------------------------------

def retry_function(fn: Callable, num_retries: int, error_msg: str='') -> Any:
    assert num_retries > 0, f'num_retries must be > 0, got {num_retries}'
    attempts = 0
    while attempts < num_retries:
        try:
            return fn()
        except Exception as e: # pylint: disable=broad-except
            attempts += 1
            print(error_msg)
            print(f'Attempt {attempts}/{num_retries} failed with error: {str(e)}')
            traceback.print_exc()
    print(f'Failed after {num_retries} attempts. Moving on.')

#----------------------------------------------------------------------------

def print_tick_stats0(stats, tick_stats, loss_mean: float, device, maintenance_time) -> None:
    fields = []
    sec_per_nimg = (tick_stats.end_time - tick_stats.start_time) / (stats.cur_nimg - tick_stats.start_nimg)
    step_per_sec = (stats.cur_step - tick_stats.start_step) / (tick_stats.end_time - tick_stats.start_time)

    fields += [f'[{time.strftime("%Y-%m-%d %H:%M")}]']
    fields += [f"step {stats.cur_step:<7d}"]
    fields += [f"tick {stats.cur_tick:<5d}"]
    fields += [f"kimg {stats.cur_nimg / 1e3:<9.1f}"]
    fields += [f"time {format_time(tick_stats.end_time - stats.start_time):<12s}"]
    fields += [f"iter/sec {step_per_sec:<7.2f}"]
    fields += [f"sec/kimg/gpu {sec_per_nimg * 1e3 * dist.get_world_size():<7.2f}"]
    fields += [f"samples/sec {1.0 / (sec_per_nimg + 1e-16):<7.3f}"]
    fields += [f"samples/sec/gpu {1.0 / (sec_per_nimg + 1e-16) / dist.get_world_size():<7.3f}"]
    fields += [f"maintenance {maintenance_time:<6.1f}"]
    fields += [f"cpumem {psutil.Process(os.getpid()).memory_info().rss / 2**30:<6.2f}"]
    fields += [f"gpumem {torch.cuda.max_memory_allocated(device) / 2**30:<6.2f}"]
    fields += [f"reserved {torch.cuda.max_memory_reserved(device) / 2**30:<6.2f}"]
    fields += [f"loss {loss_mean:<4.3f}"]

    dist.print0(' '.join(fields))

#----------------------------------------------------------------------------

def print_training_options(cfg, data: Data, resumed_snapshot: os.PathLike=None):
    # Print options.
    dist.print0()
    dist.loginfo0(f'Experiment directory:    {cfg.experiment_dir}')
    dist.loginfo0(f'Sample shape:            {cfg.dataset.resolution}')
    dist.loginfo0(f'Label shape:             {cfg.dataset.label_shape}')
    dist.loginfo0(f'Network architecture:    {cfg.model.arch}')
    dist.loginfo0(f'Preconditioning:         {cfg.model.precond}')
    dist.loginfo0(f'Loss:                    {cfg.loss.name}')
    dist.loginfo0(f'Number of GPUs:          {dist.get_world_size()}')
    dist.loginfo0(f'Number of nodes:         {dist.get_num_nodes()}')
    dist.loginfo0(f'Activations dtype:       {cfg.model.act_dtype}')
    dist.loginfo0(f'Params dtype:            {cfg.model.param_dtype}')
    dist.loginfo0(f'Resumed snapshot:        {resumed_snapshot}')
    dist.print0()
    if isinstance(data.dataloader, torch.utils.data.DataLoader):
        dist.loginfo0(f'Batch size:            {data.batch_info.batch_size}')
        dist.loginfo0(f'Batch per GPU:         {data.batch_info.batch_gpu}')
        dist.loginfo0(f'Num accum rounds:      {data.batch_info.num_accum_rounds}')
        dist.loginfo0(f'Epoch size:            {len(data.dataset)}')
        dist.loginfo0(f'Data type:             {data.dataset.data_type}')
        dist.loginfo0(f'Resolution:            {data.dataset.resolution}')
    dist.print0()

#----------------------------------------------------------------------------
# Utils to compute and log parameter statistics.

@torch.no_grad()
@misc.profiled_function
def log_param_stats(modules: dict[str, Optional[torch.nn.Module]], stats) -> None:
    dist.loginfo0("Logging parameter statistics...", end='')
    log_dict = {}
    for module_name, module in modules.items():
        if module is None: continue
        # for var_type, var_generator in {'params': module.named_parameters(), 'buffers': module.named_buffers()}.items():
        for var_type, var_generator in {'params': module.named_parameters()}.items():
            for param_name, parameter in var_generator:
                param_and_grad_stats = compute_param_and_grad_stats(parameter)
                for stats_type, stats_values in param_and_grad_stats.items():
                    if stats_values is None: continue
                    for statistic_name, statistic_value in stats_values.items():
                        log_dict[f'{module_name}_{var_type}_{stats_type}/{statistic_name}/{param_name}'] = statistic_value

    wandb_log_with_stats0(log_dict, stats, commit=True)
    dist.loginfo0("done!")


@torch.no_grad()
def log_param_update_stats(modules: dict[str, Optional[torch.nn.Module]], params_prev_copy: dict[str, torch.Tensor], stats) -> None:
    dist.loginfo0("Logging parameter update statistics...", end='')
    log_dict = {}
    for module_name, module in modules.items():
        if module is None or not module_name in params_prev_copy: continue
        for param_name, parameter_cur in module.named_parameters():
            parameter_prev = params_prev_copy[module_name][param_name]
            param_prev_stats = compute_param_and_grad_stats(parameter_prev).param
            update_stats = compute_param_and_grad_stats(parameter_cur.cpu().double() - parameter_prev.double()).param
            param_update_ratio_stats = EasyDict(**{k: (update_stats[k] / (param_prev_stats[k] + 1e-8)) for k in update_stats.keys()})

            for stats_type, stats_values in dict(update=update_stats, param_update_ratio=param_update_ratio_stats).items():
                if stats_values is None: continue
                for statistic_name, statistic_value in stats_values.items():
                    log_dict[f'{module_name}_{stats_type}_{statistic_name}/{param_name}_update'] = statistic_value.float().item()

    wandb_log_with_stats0(log_dict, stats, commit=True)
    dist.loginfo0("done!")


def compute_param_and_grad_stats(x: torch.Tensor) -> EasyDict[str, Optional[EasyDict[str, float]]]:
    assert isinstance(x, torch.Tensor), f"Parameter is not a tensor: {type(x)}"
    x = misc.full_tensor(x) # [...]
    param_stats = EasyDict(std=x.std(), mean=x.mean(), mean_abs=x.abs().mean(), norm=x.norm())
    grad_stats = None if not x.requires_grad or x.grad is None else EasyDict(std=x.grad.std(), mean=x.grad.mean(), mean_abs=x.grad.abs().mean(), norm=x.grad.norm())
    param_grad_ratio_stats = None if not x.requires_grad or x.grad is None else EasyDict(**{k: (grad_stats[k] / (v + 1e-8)) for k, v in param_stats.items()})

    return EasyDict(param=param_stats, grad=grad_stats, param_grad_ratio=param_grad_ratio_stats)


def log_params_diff(student: torch.nn.Module, teacher: torch.nn.Module, stats: TrainingStats) -> None:
    dist.loginfo0("Logging parameter differences...", end='')
    log_dict = {}
    for pname_s, param_s in student.named_parameters():
        param_t = misc.find_node(teacher, pname_s)
        diff = param_s - param_t # [...]
        param_diff_stats = compute_param_and_grad_stats(diff).param
        for stats_type, stats_value in param_diff_stats.items():
            log_dict[f'net_teacher_params_diff_{stats_type}/{pname_s}'] = stats_value.float().item()

    wandb_log_with_stats0(log_dict, stats, commit=True)
    dist.loginfo0("done!")

#----------------------------------------------------------------------------
# Wandb logging utils.

def maybe_init_wandb(cfg: EasyDict) -> None:
    if not cfg.wandb.enabled or (not cfg.wandb.distributed and not dist.is_main_process()):
        return

    wandb_run_suffx = f'-rank-{dist.get_rank()}' if cfg.wandb.distributed else ''
    # We initialize wandb in each process so that it collects system statistics.
    wandb.init(
        project=cfg.wandb.project,
        dir=cfg.wandb.dir,
        group=cfg.experiment_name_full if cfg.wandb.distributed else cfg.dataset.name,
        config=EasyDict.init_recursively(cfg, target_class=dict),
        name=f'{cfg.experiment_name_full}{wandb_run_suffx}',
        resume='allow',
        id=f'{cfg.env.user}{wandb_run_suffx}',
        tags=[
            cfg.env.user,
            cfg.dataset.name,
            cfg.model.arch,
            cfg.model.act_dtype,
            f'rank-{dist.get_rank()}',
            'x'.join([str(r) for r in cfg.dataset.resolution]),
        ] + cfg.wandb.tags + (['rank-main'] if dist.is_main_process() else []),
    )

def wandb_log_with_stats(log_dict: dict, stats: TrainingStats, **wandb_log_kwargs) -> None:
    log_dict.update({'Progress/nimg': stats.cur_nimg, 'Progress/step': stats.cur_step})
    wandb.log(log_dict, step=stats.cur_step, **wandb_log_kwargs)

def wandb_log_with_stats0(*args, **kwargs) -> None:
    if dist.is_main_process():
        wandb_log_with_stats(*args, **kwargs)

def maybe_log_losses_and_grads(cfg: EasyDict, stats: TrainingStats, loss_means_per_lvl: EasyDict, net_grad_log_dict: EasyDict, grad_norm_log_dict: EasyDict, **wandb_log_kwargs) -> None:
    if not cfg.wandb.enabled:
        return
    loss_dict = {f'Loss/train_{k}': v.mean() for k, v in loss_means_per_lvl.items()}
    for adv_loss_key in [k for k in loss_dict if '_adv_' in k or '_discr_' in k]:
        new_adv_loss_key = adv_loss_key.replace('Loss/', '').replace('_adv_', '_')
        loss_dict[f'Loss_adv/{new_adv_loss_key}'] = loss_dict[adv_loss_key]
        del loss_dict[adv_loss_key]
    # loss_dict = {k.replace('Loss/', 'Loss_adv/').replace('adv_', '') if '_adv_' in k else k: v for k, v in loss_dict.items()} # Rename adversarial losses for better visualization.
    loss_log_fn = wandb_log_with_stats if cfg.wandb.distributed else wandb_log_with_stats0
    log_dict = {**loss_dict, **net_grad_log_dict, **grad_norm_log_dict}
    loss_log_fn(log_dict, stats, **wandb_log_kwargs)

#----------------------------------------------------------------------------

def init_custom_logger(output_dir: os.PathLike) -> None:
    if dist.is_main_process():
        Logger(file_name=os.path.join(output_dir, 'log.txt'), file_mode='a', should_flush=True)

#----------------------------------------------------------------------------

def log_autoencoder_stats(stats_dict: EasyDict[str, BasicStats], training_stats: TrainingStats, save_to_wandb: bool=False, wandb_prefix: str='') -> None:
    for prefix, stats in stats_dict.items():
        stats_agg = BasicStatsAgg.infer_from_basic_stats(stats)
        dist.loginfo0(f'Computed {wandb_prefix}/{prefix} stats! {stats_agg}')

        if save_to_wandb:
            stats_to_wandb_log = {f'{wandb_prefix}/{prefix}/{k}': v for k, v in asdict(stats_agg).items()}
            wandb_log_with_stats0(stats_to_wandb_log, training_stats)

#----------------------------------------------------------------------------

@torch.no_grad()
def print_model_info(cfg, batch_gpu: int, nets: Nets, device: str | torch.device):
    batch_gpu = min(batch_gpu, 2) # Do not use a batch size larger than 2 since print_module_summary can take extra memory on storing the activations.
    if cfg.model.get('is_autoencoder') or cfg.model.get('is_lgm'):
        autoencoder_model = misc.unwrap_module(nets.net).model if cfg.model.get('is_autoencoder') else misc.unwrap_module(nets.net).autoencoder.model
        dist.print0(f'Latent resolution: {"x".join(map(str, autoencoder_model.latent_resolution))}. Latent channels: {autoencoder_model.latent_channels}.')

    dummy_inputs = misc.unwrap_module(nets.net).construct_dummy_input(batch_gpu, device) # [b, t, c, h, w], [b], [b, ...]
    # We run `misc.print_module_summary` on each device since the first forward pass can change the strides of parameter tensors in pytorch 2.
    # If this happens, then DDP fails since it expects all the parameters' sizes/strides to be the same.
    misc.print_module_summary(nets.net, dummy_inputs, max_nesting=3, verbose=dist.is_main_process())
    if cfg.model.is_lgm:
        ae_dummy_inputs = misc.unwrap_module(nets.net).autoencoder.construct_dummy_input(batch_gpu, device, input_shape=[cfg.dataset.resolution[0], 3, *cfg.dataset.resolution[1:]]) # [b, t, c, h, w], [b], [b, ...]
        misc.print_module_summary(
            misc.unwrap_module(nets.net).autoencoder, ae_dummy_inputs, max_nesting=3, verbose=dist.is_main_process(),
            module_kwargs=dict(encode=True, decode=True, force_label_dropout=cfg.model.autoencoder_ignore_labels),
        )

    # Printing the model as well to better see its internals.
    dist.print0(nets.net)

#----------------------------------------------------------------------------

def log_real_samples(cfg, nets, vis, stats, device):
    if vis.get('videos') is None:
        return

    if cfg.model.is_lgm:
        # Decoding or encoding/decoding the real videos to log. Saving the reconstructions instead of RGB since it gives a much better understanding of what we train on.
        with torch.no_grad():
            if cfg.model.use_precomputed_latents:
                latents = vis.videos.float().to('cuda') # [num_samples, num_frames, c, h, w]
            else:
                videos_real_gt = misc.maybe_rescale_uint8_to_fp32(vis.videos.to(device)) # [num_samples, num_frames, c, h, w]
                latents = torch.cat([misc.unwrap_module(nets.net).autoencoder(videos_real_gt[i:i+1], sigma=None, cond=None, encode=True, decode=False, return_extra_output=True)[1]['latents'] for i in range(videos_real_gt.shape[0])], dim=0) # [num_samples, num_frames, c, h, w]
            ae_sampling_kwargs = dict(cfg=EasyDict(**{**misc.unwrap_module(nets.net).autoencoder.cfg.sampling, **dict(real_init=False)}), net=misc.unwrap_module(nets.net).autoencoder, encode=False, device=device)
            videos_real = torch.cat([sample_batch_diffusion(**ae_sampling_kwargs, batch_seeds=torch.as_tensor([i], device=device), cond=None, latents=l.unsqueeze(0)).squeeze(0) for i, l in enumerate(latents)], dim=0) # [b, t, c, h, w]
            videos_real = misc.maybe_rescale_fp32_to_uint8(videos_real.cpu()) # [num_samples, t, c, h, w]
    else:
        videos_real = vis.videos # [num_samples, t, c, h, w]
    videos_real = einops.rearrange(videos_real, 'b t c h w -> b 1 t h w c') # [num_samples, 1, t, h, w, c]
    samples_real = TensorGroup(misc.filter_nones(dict(video=videos_real, audio=vis.audio))) # [num_samples, 1, t, h, w, c], [num_samples, ca, ta] or None
    save_samples(samples_real, stats, cfg.output_dir, save_name='real', save_to_wandb=cfg.wandb.enabled, as_grid=vis.save_videos_as_grid)
    vis.videos = vis.audio = None # Cleaning up so that it does not leak into the state dict.

#----------------------------------------------------------------------------

class Logger(object):
    """Redirect stderr to stdout, optionally print stdout to a file, and optionally force flushing on both stdout and the file."""

    def __init__(self, file_name: Optional[str] = None, file_mode: str = "w", should_flush: bool = True):
        self.file = None

        if file_name is not None:
            self.file = open(file_name, file_mode)

        self.should_flush = should_flush
        self.stdout = sys.stdout
        self.stderr = sys.stderr

        sys.stdout = self
        sys.stderr = self

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def write(self, text: Union[str, bytes]) -> None:
        """Write text to stdout (and a file) and optionally flush."""
        if isinstance(text, bytes):
            text = text.decode()
        if len(text) == 0: # workaround for a bug in VSCode debugger: sys.stdout.write(''); sys.stdout.flush() => crash
            return

        if self.file is not None:
            self.file.write(text)

        self.stdout.write(text)

        if self.should_flush:
            self.flush()

    def flush(self) -> None:
        """Flush written text to both stdout and a file, if open."""
        if self.file is not None:
            self.file.flush()

        self.stdout.flush()

    def close(self) -> None:
        """Flush, close possible files, and remove stdout/stderr mirroring."""
        self.flush()

        # if using multiple loggers, prevent closing in wrong order
        if sys.stdout is self:
            sys.stdout = self.stdout
        if sys.stderr is self:
            sys.stderr = self.stderr

        if self.file is not None:
            self.file.close()
            self.file = None

#----------------------------------------------------------------------------
