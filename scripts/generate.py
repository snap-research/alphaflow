import os
import sys

import numpy as np
import torch
from torch.distributed.elastic.multiprocessing.errors import record
import hydra
from omegaconf import DictConfig
import einops

from src.structs import TensorLike, TensorGroup, EasyDict
from src.utils.os_utils import lean_tqdm, disable_trivial_warnings, parse_int_list
from src.utils import misc, logging_utils, distributed as dist
from src.inference import sample
from src.data import Data, construct_full_cond_from_batch
from src.data.utils import VIDEO_EXTENSIONS
from src.training.network_utils import load_snapshot
from src.utils.autoencoder_utils import init_autoencoder_stats, compute_autoencoder_stats
from infra.utils import recursive_instantiate

#----------------------------------------------------------------------------

@record
@hydra.main(config_path="../configs", config_name="generate.yaml", version_base='1.2')
def generate(cfg: DictConfig):
    cfg = EasyDict.init_recursively(cfg)
    disable_trivial_warnings()
    assert cfg.output_dir is not None, 'Please specify output_dir=<path/to/outputs>'
    if cfg.distributed:
        dist.init()
    device = torch.device('cuda')
    dist.init_random_state_and_cuda(seed=42, cudnn_benchmark=cfg.cudnn_benchmark, allow_tf32=cfg.allow_tf32)

    seeds = parse_int_list(cfg.seeds) # [num_seeds]
    num_batches = ((len(seeds) - 1) // (cfg.batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_seeds = all_batches[dist.get_rank() :: dist.get_world_size()]

    assert len(seeds) % dist.get_world_size() == 0, f'Number of seeds ({len(seeds)}) must be divisible by the number of GPUs ({dist.get_world_size()})'
    assert all(len(seeds) > 0 for seeds in rank_seeds), f'No seeds assigned to rank ({dist.get_rank()}): {rank_seeds}'

    if cfg.profiling.enabled and (dist.is_main_process() or cfg.profiling.trace_all_ranks):
        profiler = misc.init_profiler(cfg.profiling.logs_dir)
        os.makedirs(cfg.profiling.logs_dir, exist_ok=True)
        dist.loginfo0(f"Profiler is enabled. Logs will be saved to {cfg.profiling.logs_dir}")
        profiler.start()
    else:
        profiler = None

    net, snapshot_path, experiment_cfg = load_snapshot(cfg.ckpt, verbose=cfg.verbose and dist.is_main_process(), device=device)
    net = net.eval().to(device)
    if 'sampling' not in cfg:
        cfg.sampling = experiment_cfg.sampling
    net.cfg.sampling = experiment_cfg.sampling

    if 'dataset' in cfg:
        # Use batching/resolution from the original dataset, but everything else from the new one.
        cfg.dataset.resolution = experiment_cfg.dataset.resolution
        cfg.dataset.predownload = experiment_cfg.dataset.predownload
        experiment_cfg.dataset = EasyDict.init_recursively(cfg.dataset).to_dict()

    recursive_instantiate(experiment_cfg)
    experiment_cfg = EasyDict.init_recursively(experiment_cfg)

    if cfg.save_latents_num_samples > 0:
        latents_path = os.path.join(cfg.env.latents_stats_dir, f'{experiment_cfg.experiment_id_str if cfg.save_latents_prefix is None else cfg.save_latents_prefix}-{os.path.basename(snapshot_path)}-{experiment_cfg.dataset.name}-{"x".join(map(str, experiment_cfg.dataset.resolution))}-n1024.pkl')
        dist.loginfo0(f'Saving latents (full) to {latents_path}')
        stats, _was_loaded = init_autoencoder_stats(experiment_cfg, keys=['latents'])
        stats['latents'].capture_mean_cov = False
        stats['latents'].capture_all = True
        experiment_cfg.dataset.predownload = 0
        # If video streams are present, use only the first one. If not, use the first image stream.
        experiment_cfg.dataset.video_streams = [experiment_cfg.dataset.video_streams[0]] if len(experiment_cfg.dataset.video_streams) > 0 else []
        experiment_cfg.dataset.image_streams = [] if len(experiment_cfg.dataset.video_streams) > 0 else [experiment_cfg.dataset.image_streams[0]]
        # experiment_cfg.dataset.video_streams[0].path_local = experiment_cfg.dataset.video_streams[0].path_local.replace('data', '/lssd/data2')
        data = Data.init_from_cfg(experiment_cfg.dataset)
        compute_autoencoder_stats(net, data, experiment_cfg, device, num_samples=cfg.save_latents_num_samples, stats=stats, verbose=True) # [t * c * h * w]
        stats['latents'].save(latents_path)
        dist.loginfo0(f'Latents saved to {latents_path}')

    if len(net.label_shape) == 1 and not cfg.use_gt_labels:
        if cfg.classes is None:
            all_classes = sample_c_from_seeds(seeds, net.label_shape[0], device=device) # [num_seeds, c_dim]
        else:
            classes = torch.as_tensor(parse_int_list(cfg.classes)) # [clases_list_len]
            classes = classes.repeat((len(seeds) + len(classes) - 1) // len(classes))[:len(seeds)] # [num_seeds]
            all_classes = c_idx_to_c(classes, net.label_shape[0], device=device) # [num_seeds, c_dim]
        labels = all_classes.tensor_split(num_batches)[dist.get_rank() :: dist.get_world_size()] # (num_batches, [num_seeds, c_dim])
    else:
        labels = [None] * len(rank_seeds)

    load_x_cond: bool = misc.unwrap_module(net).cfg.use_x_cond or cfg.save_gt
    load_labels: bool = len(net.label_shape) > 1 or labels[0] is None
    if load_labels or load_x_cond:
        assert cfg.classes is None, 'Cannot specify --classes when upsampling is enabled'
        if cfg.selected_stream is not None: Data.filter_stream_in_cfg_(experiment_cfg.dataset, cfg.selected_stream)
        if cfg.get('new_dataset_shuffle_seed') is not None: experiment_cfg.training.seed = cfg.new_dataset_shuffle_seed
        data = Data.init_from_cfg(experiment_cfg.dataset)
        iterator = data.iterator
    else:
        iterator = None

    # Loop over batches.
    if cfg.verbose:
        dist.loginfo0(f'Generating {len(torch.cat(rank_seeds))} videos on each of {dist.get_world_size()} GPUs to "{cfg.output_dir}"...')

    if cfg.skip_if_exists and os.path.isdir(cfg.output_dir):
        import PIL.Image; PIL.Image.init()
        existing_files = [f for f in os.listdir(cfg.output_dir) if os.path.isfile(os.path.join(cfg.output_dir, f))]
        existing_media_files = [f for f in existing_files if any(f.endswith(ext) for ext in (list(VIDEO_EXTENSIONS) + list(PIL.Image.EXTENSION.keys())))]
        if len(existing_media_files) == len(seeds):
            dist.loginfo0(f'Skipping the generatioin process since all samples already exist in "{cfg.output_dir}"')
            sys.exit(0)

    if cfg.save_traj_step_idx is not None:
        assert cfg.save_traj_len is None or len(cfg.save_traj_step_idx) == cfg.save_traj_len, f'Length of `save_traj_step_idx` ({len(cfg.save_traj_step_idx)}) must be equal to `save_traj_len` ({cfg.save_traj_len})'
    common_kwargs = dict(verbose=False, device=device, save_traj_len=cfg.save_traj_len, save_traj_step_idx=cfg.save_traj_step_idx)
    for cur_seeds, cur_labels in lean_tqdm(zip(rank_seeds, labels), total=len(rank_seeds), desc='Generating videos', disable=not cfg.verbose or not dist.is_main_process()):
        with torch.profiler.record_function('data_fetch'):
            batch_real = None if iterator is None else {k: v[:len(cur_seeds)] for k, v in next(iterator).items()} # [batch_size, ...]
            cur_cond = None if batch_real is None else construct_full_cond_from_batch(batch_real, load_x_cond) # [num_seeds, ...]
            if len(net.label_shape) == 1:
                if cur_cond is None:
                    cur_cond = net.construct_cond_from_labels(cur_labels) # [num_seeds, ...]
                else:
                    cur_cond['label'] = cur_labels
            if cfg.overwrite_fps is not None:
                cur_cond['framerate'].fill_(cfg.overwrite_fps)
            if cfg.overwrite_dataset_id is not None:
                cur_cond['dataset_id'].fill_(cfg.overwrite_dataset_id)
            if cfg.overwrite_video_flag is not None:
                cur_cond['is_video'].fill_(cfg.overwrite_video_flag)
            if cfg.overwrite_resolution is not None:
                dist.loginfo0(f'Resolution before {cur_cond["resolution"]}')
                cur_cond['resolution'] = torch.as_tensor(cfg.overwrite_resolution, dtype=cur_cond['resolution'].dtype, device=cur_cond['resolution'].device).expand_as(cur_cond['resolution']) # [batch_size, 2]
                dist.loginfo0(f'Resolution after {cur_cond["resolution"]}')
            captions = ([None] * len(cur_seeds)) if batch_real is None or not 'text' in batch_real else batch_real['text'] # [batch_size, <text>]

        samples: TensorLike = sample(net, cfg.sampling, [cur_seeds], [cur_cond], **common_kwargs, **cfg.model_kwargs) # [num_seeds, traj_len, t, c, h, w]
        samples = TensorGroup(video=samples) if not isinstance(samples, TensorGroup) else samples

        with torch.profiler.record_function('data_saving'):
            logging_utils.save_samples(
                samples=samples,
                stats=None,
                output_dir=cfg.output_dir,
                save_name='sample',
                save_to_wandb=False,
                seeds=cur_seeds.tolist(),
                use_res_suffix=False,
                captions=captions,
                **cfg.save_opts,
            )

            if cfg.save_gt:
                for sample_idx, (seed, caption) in enumerate(zip(cur_seeds, captions)):
                    output_dir_gt = f'{cfg.output_dir}-gt'
                    os.makedirs(output_dir_gt, exist_ok=True)
                    for seed, gt_video in zip(cur_seeds, cur_cond.x_cond):
                        gt_video = misc.maybe_rescale_fp32_to_uint8(einops.rearrange(gt_video, 't c h w -> t h w c')).cpu().numpy() # [t, h, w, c]
                        logging_utils.save_video(gt_video, os.path.join(output_dir_gt, f'{seed.item():06d}'), **cfg.save_opts)
        if not profiler is None:
            profiler.step()

    # Done.
    if not profiler is None:
        profiler.stop()
    dist.barrier()
    if cfg.verbose:
        dist.loginfo0('Done.')
    sys.exit(0)

#----------------------------------------------------------------------------

def sample_c_from_seeds(seeds: list[int], c_dim: int, device: str='cpu') -> torch.Tensor:
    if c_dim == 0:
        return torch.empty(len(seeds), 0)
    c_idx = [np.random.RandomState(s).choice(np.arange(c_dim), size=1).item() for s in seeds] # [num_samples]
    return c_idx_to_c(c_idx, c_dim, device)

#----------------------------------------------------------------------------

def c_idx_to_c(c_idx: list[int], c_dim: int, device: str) -> torch.Tensor:
    c_idx = np.array(c_idx) # [num_samples, 1]
    c = np.zeros((len(c_idx), c_dim)) # [num_samples, c_dim]
    c[np.arange(len(c_idx)), c_idx] = 1.0

    return torch.from_numpy(c).float().to(device) # [num_samples, c_dim]

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
