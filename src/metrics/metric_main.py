"""
Main API for computing and reporting quality metrics.
TODO: with the current complication, we should just better parse metric args from its name...
"""

import os
import re
import time
import json
from typing import Optional, Callable
import torch

from src.utils import distributed as dist
from src.metrics import metrics
from src.utils.os_utils import format_time
from src.structs import EasyDict

#----------------------------------------------------------------------------

_metric_dict = dict() # name => fn

def register_metric(fn):
    assert callable(fn)
    _metric_dict[fn.__name__] = fn
    return fn

def is_valid_metric(metric) -> bool:
    return metric in _metric_dict

def is_multi_level_metric(metric) -> bool:
    return any(metric.startswith(prefix) for prefix in ['fid', 'fvd', 'joint', 'clip_score'])

def does_metric_require_val_data(metric: str) -> bool:
    assert is_valid_metric(metric), f'Invalid metric: {metric}'
    return any(metric.startswith(prefix) for prefix in ['clip_score', 'reconstruction'])

def list_valid_metrics():
    return list(_metric_dict.keys())

def _format_multi_level_metrics(metric_values: dict[str, float], prefix: str) -> dict[str, float]:
    # TODO: we keep the old metric name which corresponds to the highest resolution for backwards compatibility.
    # Also, it's our main metric, might be useful to have it be directly comparable between resolutions?
    main_metric_value = {prefix: metric_values[list(metric_values.keys())[-1]]}
    extra_metric_values = {f'{prefix}_{k}': metric_values[k] for k in metric_values}
    return {**main_metric_value, **extra_metric_values}

def _format_samples_count(num_samples: Optional[int]=None) -> str:
    if num_samples is None:
        return 'full'
    elif num_samples % 1000 == 0:
        return f'{num_samples // 1000}k'
    else:
        return str(num_samples)

#----------------------------------------------------------------------------

def compute_metric(
    metric: str,
    net: torch.nn.Module | None = None,
    model_kwargs: dict | None=None,
    rank: int=0,
    device: str | torch.device | None = None,
    cache: bool = True,
    verbose: bool = False,
    batch_gen: int | None = None,
    dataset_stats_dir: str | None = None,
    sampling_cfg: EasyDict=None,
    detector_batch_gpu: int = 64,
    save_partial_stats_freq = 10_000 // dist.get_world_size(),
    dataset: torch.utils.data.Dataset | None = None,
    gen_dataset: torch.utils.data.Dataset | None = None, # When a generated dataset is provided, we can use it instead of real data.
    latents_std: float | None = None, # For noised reconstruction metrics.
    audio_sample_rate: int | None = None, # For audio-based metrics, we need to pass the sample rate.
    metric_name_suffix: str | None = None, # suffix for metric name
):
    assert is_valid_metric(metric), f'Invalid metric: {metric}'
    assert dataset_stats_dir is not None, f'dataset_stats_dir must be specified, but got: {dataset_stats_dir}'

    opts = EasyDict(
        net                         = net,
        model_kwargs                = {} if model_kwargs is None else EasyDict.init_recursively(model_kwargs),
        rank                        = rank,
        device                      = device if device is not None else torch.device('cuda', rank),
        cache                       = cache,
        batch_gen                   = batch_gen, # Generator batch size
        detector_batch_gpu          = detector_batch_gpu, # Detector batch size
        sampling_cfg                = sampling_cfg,
        dataset_stats_dir           = dataset_stats_dir,
        verbose                     = verbose,
        save_partial_stats_freq     = save_partial_stats_freq,
        dataset                     = dataset,
        gen_dataset                 = gen_dataset,
        use_gen_data                = gen_dataset is not None,
        latents_std                 = latents_std,
        audio_sample_rate           = audio_sample_rate,
    )

    # Calculate.
    start_time = time.time()
    results = _metric_dict[metric](opts)
    total_time = time.time() - start_time

    # Broadcast results.
    for key, value in list(results.items()):
        if dist.get_world_size() > 1:
            value = torch.as_tensor(value, dtype=torch.float64, device=opts.device)
            torch.distributed.broadcast(tensor=value, src=0)
            value = float(value.cpu())
        results[key] = value

    # Adding the steps prefix here, and also hardcode to delete the key end with _{}x{}
    key_list = list(results.keys())
    for key in key_list:
        results[f"{key}_{metric_name_suffix}"] = results.pop(key)

    # Decorate with metadata.
    return EasyDict(
        results         = EasyDict(results),
        metric          = metric,
        total_time      = total_time,
        total_time_str  = format_time(total_time),
        world_size      = dist.get_world_size(),
    )

def report_metric(result_dict, run_dir=None, snapshot_path=None, save_result: bool=True):
    metric = result_dict['metric']
    assert is_valid_metric(metric)
    if run_dir is not None and snapshot_path is not None:
        snapshot_path = os.path.relpath(snapshot_path, run_dir)
    jsonl_line = json.dumps(dict(result_dict, snapshot_path=snapshot_path, timestamp=time.time()))
    print(jsonl_line)
    if save_result and run_dir is not None and os.path.isdir(run_dir):
        with open(os.path.join(run_dir, f'metric-{metric}.jsonl'), 'at') as f:
            f.write(jsonl_line + '\n')

#----------------------------------------------------------------------------
# FID/FVD metrics.

def make_fd_metrics(num_gen: int, max_real: Optional[int]) -> dict[str, Callable]:
    num_gen_str, max_real_str = _format_samples_count(num_gen), _format_samples_count(max_real)
    fid_metric_name = f'fid{num_gen_str}_{max_real_str}'
    dinofid_metric_name = f'dinofid{num_gen_str}_{max_real_str}'
    fvd_metric_name = f'fvd{num_gen_str}_{max_real_str}'
    fcd_metric_name = f'fcd{num_gen_str}_{max_real_str}'
    joint_fd_metric_name = f'joint_fd{num_gen_str}_{max_real_str}'
    joint_image_fd_metric_name = f'joint_image_fd{num_gen_str}_{max_real_str}'
    afibd_metric_name = f'afibd{num_gen_str}_{max_real_str}'

    def fid(opts):
        fids = metrics.compute_fid(opts, max_real=max_real, num_gen=num_gen) # (num_levels)
        return _format_multi_level_metrics(fids, fid_metric_name)

    def dinofid(opts):
        fids = metrics.compute_fid(opts, max_real=max_real, num_gen=num_gen, detector_type='dinov2') # (num_levels)
        return _format_multi_level_metrics(fids, dinofid_metric_name)

    def fvd(opts):
        fvds = metrics.compute_fvd(opts, max_real=max_real, num_gen=num_gen) # (num_levels)
        return _format_multi_level_metrics(fvds, fvd_metric_name)

    def fcd(opts):
        fcds = metrics.compute_fid(opts, max_real=max_real, num_gen=num_gen, detector_type='clip') # (num_levels)
        return _format_multi_level_metrics(fcds, fcd_metric_name)


    def joint_fd(opts):
        metric_kwargs = {
            fid_metric_name: {'kwargs': {'max_real': max_real}, 'modality': 'image'},
            dinofid_metric_name: {'kwargs': {'max_real': max_real, 'detector_type': 'dinov2'}, 'modality': 'image'},
            fcd_metric_name: {'kwargs': {'max_real': max_real, 'detector_type': 'clip'}, 'modality': 'image'},
            fvd_metric_name: {'kwargs': {'max_real': max_real}, 'modality': 'video'},
        }
        metric_values: dict[str, float] = metrics.compute_joint_frechet_distances(metric_kwargs, opts, num_gen=num_gen)
        return metric_values

    def joint_image_fd(opts):
        metric_kwargs = {
            fid_metric_name: {'kwargs': {'max_real': max_real}, 'modality': 'image'},
            dinofid_metric_name: {'kwargs': {'max_real': max_real, 'detector_type': 'dinov2'}, 'modality': 'image'},
            fcd_metric_name: {'kwargs': {'max_real': max_real, 'detector_type': 'clip'}, 'modality': 'image'},
        }
        metric_values: dict[str, float] = metrics.compute_joint_frechet_distances(metric_kwargs, opts, num_gen=num_gen)
        return metric_values

    return {
        fid_metric_name: fid,
        dinofid_metric_name: dinofid,
        fvd_metric_name: fvd,
        joint_fd_metric_name: joint_fd,
        joint_image_fd_metric_name: joint_image_fd,
        fcd_metric_name: fcd,
    }


for num_gen, max_real in [(32, 32), (128, None), (256, None), (512, None), (512, 512), (2048, None), (4096, None), (5000, None), (10_000, None), (50_000, None)]:
    _metric_dict.update(make_fd_metrics(num_gen, max_real))

#----------------------------------------------------------------------------
# CLIP Score, Reconstruction and Inception metrics.
# They are in the same section since they rely on num_gen only.

def make_synth_only_metrics(num_gen: int) -> dict[str, Callable]:
    num_gen_str = _format_samples_count(num_gen)
    is_metric_name = f'is{num_gen_str}'
    isv_metric_name = f'isv{num_gen_str}'
    cs_metric_name = f'clip_score{num_gen_str}'
    rec_metric_name = f'reconstruction{num_gen_str}'

    def isv_metric(opts):
        mean, std = metrics.compute_inception_score_video(opts, num_gen=num_gen, num_splits=10)
        return {f'{isv_metric_name}_mean': mean, f'{isv_metric_name}_std': std}

    def is_metric(opts):
        mean, std = metrics.compute_inception_score(opts, num_gen=num_gen, num_splits=10)
        return {f'{is_metric_name}_mean': mean, f'{is_metric_name}_std': std}

    def clip_score_metric(opts):
        cs_values = metrics.compute_clip_score(opts, num_gen=num_gen) # (num_levels)
        return _format_multi_level_metrics(cs_values, cs_metric_name)

    def reconstruction_metric(opts):
        return metrics.compute_reconstruction_scores(opts, num_gen=num_gen)

    return {
        isv_metric_name: isv_metric,
        is_metric_name: is_metric,
        cs_metric_name: clip_score_metric,
        rec_metric_name: reconstruction_metric,
    }

def make_lat_noised_reconstruction_metric(num_gen: int, sigma: float) -> dict[str, Callable]:
    # Reconstruction metrics with latent noise.
    num_gen_str = _format_samples_count(num_gen)
    rec_metric_name = f'reconstruction{num_gen_str}'
    lat_noise_rec_metric_name = f'lat_noise_{rec_metric_name}_sigma{str(sigma).replace(".", "_")}'

    def cur_lat_noised_reconstruction_metric(opts):
        opts.model_kwargs = dict(latents_noise_std=opts.latents_std * sigma, **opts.model_kwargs)
        rec_scores: dict[str, float] = metrics.compute_reconstruction_scores(opts, num_gen=num_gen)
        return {k + f'_lat_noise_sigma{sigma}': v for k, v in rec_scores.items()}

    return {lat_noise_rec_metric_name: cur_lat_noised_reconstruction_metric}

for num_gen in [16, 64, 128, 256, 512, 2048, 2048, 4096, 8192, 10_000, 50_000]:
    _metric_dict.update(make_synth_only_metrics(num_gen))

    for sigma in [0.1, 0.25, 0.5, 1]:
        _metric_dict.update(make_lat_noised_reconstruction_metric(num_gen, sigma))

#----------------------------------------------------------------------------

_metric_dict = {k: _metric_dict[k] for k in sorted(_metric_dict.keys())}

#----------------------------------------------------------------------------
