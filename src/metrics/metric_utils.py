import os
import math

import numpy as np
import scipy.linalg
import torch
from src.data.utils import inf_loop_dataloader

from src import data
from src.structs import TensorLike, EasyDict, TensorGroup
from src.inference import sample
from src.utils import misc, distributed as dist
from src.utils.feature_stats import FeatureStats
from src.data import construct_full_cond_from_batch
from src.utils.os_utils import lean_tqdm, upper_dividend
from src.metrics.detectors import Detector

#----------------------------------------------------------------------------

def get_cache_file_path(dataset, detector: Detector, opts, resize_frames, num_items: int, stats_type: str) -> tuple[str, str]:
    resize_str = ('' if resize_frames is None else ('-resize' + 'x'.join([str(r) for r in resize_frames])))
    cache_tag = f'{dataset.get_identifier_desc()}-n{num_items}{resize_str}-{detector.name}-stat-{stats_type}'
    cache_file = os.path.join(opts.dataset_stats_dir, f'{cache_tag}.pkl')

    return cache_file, cache_tag

#----------------------------------------------------------------------------

@torch.no_grad()
def compute_feature_stats_for_dataset(opts, detector: Detector, max_items=None, resize_frames: tuple[int, int]=None, stats_type='mean_cov') -> FeatureStats:
    assert stats_type in ['all', 'mean_cov'], f'Unknown stats_type: {stats_type}'
    detector_batch_size = min(opts.detector_batch_gpu, opts.dataset.epoch_size // dist.get_world_size())
    dataloader_kwargs = dict(pin_memory=True, num_workers=1, drop_last=True) # It's important to drop the last batch to avoid the last batch being smaller than the rest.
    dataloader = iter(torch.utils.data.DataLoader(dataset=data.get_dp_data_subset(opts.dataset), batch_size=detector_batch_size, **dataloader_kwargs))
    dataset_iterator = iter(lean_tqdm(dataloader, disable=not (opts.verbose and dist.is_main_process())))
    assert len(dataloader) > 0, f'No items in the dataset: {len(opts.dataset), len(dataloader), opts.dataset.epoch_size, detector_batch_size}'

    if opts.cache:
        # Try to lookup from cache.
        cache_file, cache_tag = get_cache_file_path(opts.dataset, detector, opts, resize_frames, min(opts.dataset.epoch_size, max_items or float('inf')), stats_type)
        is_cache_present = os.path.isfile(cache_file) if dist.is_main_process() else False

        if dist.get_world_size() > 1:
            is_cache_present = torch.as_tensor(is_cache_present, dtype=torch.float32, device=opts.device)
            torch.distributed.broadcast(tensor=is_cache_present, src=0)
            is_cache_present = float(is_cache_present.cpu()) != 0

        if is_cache_present:
            return FeatureStats.load(cache_file)
    else:
        cache_file = cache_tag = None

    dist.loginfo0(f'Computing dataset stats {cache_tag} for {opts.dataset.name}')

    # Initialize the stats to fill.
    stats = FeatureStats(max_items=max_items, capture_all=(stats_type == 'all'), capture_mean_cov=(stats_type == 'mean_cov')) # pylint: disable=superfluous-parens

    # Main loop.
    dummy_feats = None
    batch_idx = 0
    cur_dataset_state = opts.dataset.state_dict() if hasattr(opts.dataset, 'state_dict') else None
    opts.dataset.load_state_dict({'epoch': 0, 'sample_in_epoch': 0}) if hasattr(opts.dataset, 'load_state_dict') else None
    while not stats.are_all_ranks_done():
        try:
            batch = next(dataset_iterator) # [detector_batch_gpu, c, h, w]
            videos = batch['video'].to(opts.device) # [b, c, t, h, w]
            feats = detector.run(videos, resize_frames, text=batch.get('text'), waveform=batch.get('audio')) # [b, d]
            dummy_feats = feats * torch.nan # [b, d]
        except StopIteration:
            # Adding dummy features just to sync the GPUs (they will be sliced out in the stats since they are NaNs).
            assert not dummy_feats is None, 'The dataset was empty'
            feats = dummy_feats # [b, d]

        stats.set_rank_done(dist.get_rank(), feats is dummy_feats or stats.is_full())
        stats.append_torch(feats)
        batch_idx += 1
        if opts.save_partial_stats_freq is not None and batch_idx % opts.save_partial_stats_freq == 0:
            stats.safe_save_partial(cache_file)

    stats.safe_save(cache_file)
    opts.dataset.load_state_dict(cur_dataset_state) if cur_dataset_state is not None else None # Restore the dataset state.
    del detector, dataloader; misc.free_memory()
    return stats

#----------------------------------------------------------------------------

@torch.no_grad()
def compute_feature_stats_for_generator(opts, detectors: list[Detector], num_items: int, resize_frames: tuple[int, int]=None, stats_type: str='mean_cov') -> dict[str, list[FeatureStats]]:
    assert stats_type in ['all', 'mean_cov'], f'Unknown stats_type: {stats_type}'
    num_items_per_rank = math.ceil(num_items / dist.get_world_size())
    detector_batch_gpu = min(opts.detector_batch_gpu, num_items_per_rank)
    batch_gen = min(opts.batch_gen or float('inf'), detector_batch_gpu)

    # Setup the generator, feature stats and the dataset (for conditioning iteration).
    net_cfg = misc.unwrap_module(opts.net).cfg
    stats_kwargs = dict(max_items=num_items, capture_all=(stats_type == 'all'), capture_mean_cov=(stats_type == 'mean_cov')) # pylint: disable=superfluous-parens
    stats_all = {d.name: [FeatureStats(**stats_kwargs)] for d in detectors}
    opts.dataset.set_progress(epoch=0, sample_in_epoch=0)
    dataset_iterator = iter(inf_loop_dataloader(torch.utils.data.DataLoader(dataset=opts.dataset, batch_size=detector_batch_gpu, num_workers=4, pin_memory=True, prefetch_factor=2)))
    # We can initialize a bit more seeds than num_gen, but that's fine since we'll crop them in FeatureStats anyway.
    # We need this to make sure that all the GPUs have the same amount of work (they are getting stuck otherwise :|).
    num_seeds = upper_dividend(upper_dividend(num_items, dist.get_world_size()), detector_batch_gpu)
    seeds = torch.arange(num_seeds, device=opts.device)[dist.get_rank() :: dist.get_world_size()] # [max_items // world_size]
    num_batches_per_rank = math.ceil(len(seeds) / detector_batch_gpu)
    assert num_batches_per_rank > 0, f'No items to compute: {len(opts.dataset), opts.dataset.epoch_size, num_batches_per_rank, detector_batch_gpu}'

    # Main loop.
    for batch_idx in lean_tqdm(range(num_batches_per_rank), disable=not (opts.verbose and dist.is_main_process())):
        curr_seeds = seeds[batch_idx * detector_batch_gpu : (batch_idx + 1) * detector_batch_gpu] # [detector_batch_gpu]
        batch = next(dataset_iterator) # [detector_batch_gpu, c, h, w]
        cond = construct_full_cond_from_batch(batch, net_cfg.use_x_cond).to(opts.device) # [detector_batch_gpu, ...]
        cur_seeds, cond = curr_seeds[:len(cond)], cond[:len(curr_seeds)] # [detector_batch_gpu], [detector_batch_gpu, ...]
        cur_seeds_split, cond_split = cur_seeds.split(batch_gen), cond.split(batch_gen) # (num_batches, [batch_gen]), (num_batches, [batch_gen, ...])
        samples: TensorLike = sample(opts.net, opts.sampling_cfg, cur_seeds_split, cond=cond_split, verbose=False, device=opts.device, **opts.model_kwargs) # [detector_batch_gpu, t, h, w, c]
        samples_split = samples.split(detector_batch_gpu)
        x_cond_split = [misc.maybe_rescale_fp32_to_uint8(vgt) for vgt in cond.x_cond.split(detector_batch_gpu)] if cond.get('x_cond') is not None else [None] * len(samples_split)
        for x, vgt in zip(samples_split, x_cond_split):
            for d in detectors:
                video = x.video if isinstance(x, TensorGroup) and hasattr(x, 'video') else x # [b, t, c, h, w] or None
                video = misc.rearrange(video, 'b t h w c -> b t c h w').to(opts.device) if video is not None else None # [b, t, c, h, w] or None
                waveform = x.audio.to(opts.device) if isinstance(x, TensorGroup) and hasattr(x, 'audio') else None # [b, c, l] or None
                feats = d.run(video, resize_frames, text=batch.get('text'), videos_gt=vgt, waveform=waveform) # [b, d]
                stats_all[d.name][0].append_torch(feats) # [n, d]
    del detectors, dataset_iterator; misc.free_memory()
    return stats_all

#----------------------------------------------------------------------------

def rewrite_opts_for_use_gen_data(opts):
    """
    Updates dataset arguments in the opts to enable the second dataset stats computation
    """
    new_opts = EasyDict(**opts)
    new_opts.dataset = new_opts.gen_dataset
    new_opts.cache = False

    return new_opts

#----------------------------------------------------------------------------

def compute_frechet_distance(mean_gen, cov_gen, mean_real, cov_real) -> float:
    assert np.all(cov_gen == cov_gen.T), f'Generator covariance matrix is not symmetric: {cov_gen}'
    assert np.all(cov_real == cov_real.T), f'Real covariance matrix is not symmetric: {cov_real}'
    if not scipy.__version__.startswith('1.10'):
        dist.logwarn0(f'Frechet distance computation can be broken for scipy versions other than scipy==1.10 (you have {scipy.__version__})')
    m = np.square(mean_gen - mean_real).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(cov_gen, cov_real), disp=False) # pylint: disable=no-member
    fid = np.real(m + np.trace(cov_gen + cov_real - s * 2))
    return float(fid)

#----------------------------------------------------------------------------
