import numpy as np
from src.metrics import metric_utils, detectors
from src.structs import EasyDict

#----------------------------------------------------------------------------
# Frechet Inception Distance (FID) from the paper
# "GANs trained by a two time-scale update rule converge to a local Nash
# equilibrium". Matches the original implementation by Heusel et al. at
# https://github.com/bioinf-jku/TTUR/blob/master/fid.py

def prepare_image_fd_stats_and_detector(opts, max_real, detector_type: str='inception') -> tuple[list[metric_utils.FeatureStats], list[str], metric_utils.Detector]:
    w_res_list = [int(opts.dataset.resolution[2])]
    h_res_list = [int(opts.dataset.resolution[1])]
    resolution_keys = [f'1x{min(h_res, w_res)}x{min(h_res, w_res)}' for h_res, w_res in zip(h_res_list, w_res_list)]

    if detector_type == 'inception':
        detector = detectors.construct_inception_detector(opts.device, opts.verbose)
    elif detector_type == 'dinov2':
        detector = detectors.construct_dinov2_detector(opts.device, opts.verbose)
    elif detector_type == 'clip':
        detector = detectors.construct_clip_detector(opts.device, opts.verbose, return_embeddings='image')
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")

    real_stats_all = []
    for h_res, w_res in zip(h_res_list, w_res_list):
        real_stats = metric_utils.compute_feature_stats_for_dataset(
            opts=opts, detector=detector, resize_frames=(min(h_res, w_res), min(h_res, w_res)), stats_type='mean_cov', max_items=max_real)
        real_stats_all.append(real_stats.get_mean_cov())

    return real_stats_all, resolution_keys, detector


def compute_frechet_distances(real_stats_all: list[metric_utils.FeatureStats], gen_stats_all: list[metric_utils.FeatureStats], resolution_keys: list[str]) -> dict[str, float]:
    frechet_distances = {}
    for lvl_idx, resolution_key in enumerate(resolution_keys):
        mean_gen, cov_gen = gen_stats_all[lvl_idx].get_mean_cov()
        mean_real, cov_real = real_stats_all[lvl_idx]
        frechet_distances[resolution_key] = metric_utils.compute_frechet_distance(mean_gen, cov_gen, mean_real, cov_real)
    return frechet_distances


def compute_fid(opts, max_real: int, num_gen: int, detector_type: str='inception') -> dict[int, float]:
    real_stats_all, resolution_keys, detector = prepare_image_fd_stats_and_detector(opts, max_real, detector_type=detector_type)

    if opts.use_gen_data:
        gen_opts = metric_utils.rewrite_opts_for_use_gen_data(opts)
        gen_stats = metric_utils.compute_feature_stats_for_dataset(gen_opts, detector, stats_type='mean_cov', max_items=num_gen)
        gen_stats_all = [gen_stats]
    else:
        gen_stats_all = metric_utils.compute_feature_stats_for_generator(opts, [detector], num_items=num_gen, stats_type='mean_cov')[detector.name]

    if opts.rank != 0:
        return {k: float('nan') for k in resolution_keys}

    return compute_frechet_distances(real_stats_all, gen_stats_all, resolution_keys)

#----------------------------------------------------------------------------
# Inception Score (IS) from the paper "Improved techniques for training
# GANs". Matches the original implementation by Salimans et al. at
# https://github.com/openai/improved-gan/blob/master/inception_score/model.py

def compute_inception_score(opts, num_gen, num_splits):
    detector = detectors.construct_legacy_inception_detector(opts.device, opts.verbose)

    if opts.use_gen_data:
        gen_opts = metric_utils.rewrite_opts_for_use_gen_data(opts)
        gen_probs = metric_utils.compute_feature_stats_for_dataset(gen_opts, detector, stats_type='all', max_items=num_gen).get_all()
    else:
        gen_probs = metric_utils.compute_feature_stats_for_generator(opts, [detector], num_items=num_gen, stats_type='all')[detector.name].get_all()

    if opts.rank != 0:
        return float('nan'), float('nan')

    scores = []
    for i in range(num_splits):
        part = gen_probs[i * num_gen // num_splits : (i + 1) * num_gen // num_splits]
        kl = part * (np.log(part) - np.log(np.mean(part, axis=0, keepdims=True)))
        kl = np.mean(np.sum(kl, axis=1))
        scores.append(np.exp(kl))
    return float(np.mean(scores)), float(np.std(scores))

#----------------------------------------------------------------------------
# CLIP Score metric.

def compute_clip_score(opts, num_gen: int) -> dict[str, float]:
    w_res_list = [int(opts.dataset.resolution[2])]
    h_res_list = [int(opts.dataset.resolution[1])]
    metric_keys = [f'1x{min(h_res, w_res)}x{min(h_res, w_res)}' for h_res, w_res in zip(h_res_list, w_res_list)]
    detector = detectors.construct_clip_detector(opts.device, opts.verbose)

    if opts.use_gen_data:
        gen_opts = metric_utils.rewrite_opts_for_use_gen_data(opts)
        similarities_list = [metric_utils.compute_feature_stats_for_dataset(gen_opts, detector, max_items=num_gen, stats_type='all')] # (1, [num_gen, 1])
    else:
        similarities_list = metric_utils.compute_feature_stats_for_generator(opts, [detector], num_items=num_gen, stats_type='all')[detector.name] # (num_levels, [num_gen, 1])

    similarities = {k: (100.0 * s.get_all().mean().item() if opts.rank == 0 else float('nan')) for k, s in zip(metric_keys, similarities_list)}

    return similarities

#----------------------------------------------------------------------------
# Frechet Video Distance (FVD). Matches the original tensorflow implementation from
# https://github.com/google-research/google-research/blob/master/frechet_video_distance/frechet_video_distance.py
# up to the upsampling operation. Note that this tf.hub I3D model is different from the one released in the I3D repo.

def prepare_video_fd_stats_and_detector(opts, max_real: int=None):
    t_res_list = [int(opts.dataset.resolution[0])]
    h_res_list = [int(opts.dataset.resolution[1])]
    w_res_list = [int(opts.dataset.resolution[2])]
    resolution_keys = [f'{t_res}x{min(h_res, w_res)}x{min(h_res, w_res)}' for t_res, h_res, w_res in zip(t_res_list, h_res_list, w_res_list)]
    real_stats_all = []

    detector_batch_gpu = opts.detector_batch_gpu
    detector = detectors.construct_inception3d_detector(opts.device, opts.verbose)

    for t_res, h_res, w_res in zip(t_res_list, h_res_list, w_res_list):
        opts.detector_batch_gpu = max(detector_batch_gpu // max(t_res // 16, 1), 1)

        mean_real, cov_real = metric_utils.compute_feature_stats_for_dataset(
            opts=opts, detector=detector, resize_frames=(min(h_res, w_res), min(h_res, w_res)),
            stats_type='mean_cov', max_items=max_real).get_mean_cov()
        real_stats_all.append((mean_real, cov_real))

    return real_stats_all, resolution_keys, detector

def compute_fvd(opts, max_real: int, num_gen: int):
    real_stats_all, resolution_keys, detector = prepare_video_fd_stats_and_detector(opts, max_real)
    if opts.use_gen_data:
        gen_opts = metric_utils.rewrite_opts_for_use_gen_data(opts)
        gen_stats_all = [metric_utils.compute_feature_stats_for_dataset(gen_opts, detector, stats_type='mean_cov', max_items=num_gen)]
    else:
        gen_stats_all = metric_utils.compute_feature_stats_for_generator(opts, [detector], num_items=num_gen, stats_type='mean_cov')[detector.name]

    if opts.rank != 0:
        return {k: float('nan') for k in resolution_keys}

    return compute_frechet_distances(real_stats_all, gen_stats_all, resolution_keys)

#----------------------------------------------------------------------------
# Inception Score Video metric.

def compute_inception_score_video(opts, num_gen, num_splits):
    detector = detectors.construct_video_inception_detector(opts.device, opts.verbose)
    if opts.use_gen_data:
        gen_opts = metric_utils.rewrite_opts_for_use_gen_data(opts)
        gen_probs = metric_utils.compute_feature_stats_for_dataset(gen_opts, detector, stats_type='all', max_items=num_gen).get_all()
    else:
        # gen_stats_all = metric_utils.compute_feature_stats_for_generator(opts, [detector], num_items=num_gen, stats_type='mean_cov')[detector.name]
        gen_probs = metric_utils.compute_feature_stats_for_generator(opts, [detector], num_items=num_gen, stats_type='all')[detector.name].get_all()

    if opts.rank != 0:
        return float('nan'), float('nan')

    assert len(gen_probs) > 0, f"Got empty generator output for {opts.num_gen} items."

    scores = []
    for i in range(num_splits):
        part = gen_probs[i * num_gen // num_splits : (i + 1) * num_gen // num_splits]
        kl = part * (np.log(part) - np.log(np.mean(part, axis=0, keepdims=True)))
        kl = np.mean(np.sum(kl, axis=1))
        scores.append(np.exp(kl))
    return float(np.mean(scores)), float(np.std(scores))

#----------------------------------------------------------------------------
# Joint metrics computation.

_prepare_fd_stats_and_detector_dict = {
    'image': prepare_image_fd_stats_and_detector,
    'video': prepare_video_fd_stats_and_detector,
}

def compute_joint_frechet_distances(metrics: dict[str, dict], opts, num_gen: int, **model_kwargs):
    real_stats_all_list, metric_keys_list, detectors = zip(*[_prepare_fd_stats_and_detector_dict[v['modality']](opts, **v['kwargs']) for v in metrics.values()])
    opts = EasyDict(**opts) # shallow copy
    gen_stats_all: dict[str, list[metric_utils.FeatureStats]] = metric_utils.compute_feature_stats_for_generator(opts, detectors, num_items=num_gen, stats_type='mean_cov', **model_kwargs)

    metric_values = {}
    for m_name, real_stats_all, d, metric_keys in zip(metrics.keys(), real_stats_all_list, detectors, metric_keys_list):
        curr_metrics = compute_frechet_distances(real_stats_all, gen_stats_all[d.name], metric_keys) if opts.rank == 0 else {k: float('nan') for k in metric_keys}
        main_metric_value = None
        for resolution_key, metric_value in curr_metrics.items():
            main_metric_value = metric_value
        metric_values[m_name] = main_metric_value # Adding the last metric value for the highest resolution.

    return metric_values

#----------------------------------------------------------------------------
# AutoEncoder reconstruction metrics.

_AE_RECONSTRUCTION_NAMES = ['L1', 'L2', 'LPIPS_AlexNet', 'LPIPS_VGG', 'SSIM', 'PSNR']

def compute_reconstruction_scores(opts, num_gen: int) -> dict[str, float]:
    t_res_list = [int(opts.dataset.resolution[0])]
    h_res_list = [int(opts.dataset.resolution[1])]
    w_res_list = [int(opts.dataset.resolution[2])]
    resolution_keys = [f'{t_res}x{min(h_res, w_res)}x{min(h_res, w_res)}' for t_res, h_res, w_res in zip(t_res_list, h_res_list, w_res_list)]
    detector = detectors.construct_reconstruction_detector(opts.device)

    if opts.use_gen_data:
        gen_opts = metric_utils.rewrite_opts_for_use_gen_data(opts)
        scores_list = [metric_utils.compute_feature_stats_for_dataset(gen_opts, detector, max_items=num_gen, stats_type='all').get_all()] # (1, [num_gen, num_rec_scores])
    else:
        scores_list = metric_utils.compute_feature_stats_for_generator(opts, [detector], num_items=num_gen, stats_type='all')[detector.name] # (num_levels, FeatureStats)
        scores_list = [s.get_all() for s in scores_list] # (num_levels, [num_gen, num_rec_scores])

    assert len(scores_list) == len(resolution_keys), f"Expected {len(resolution_keys)} scores, but got {len(scores_list)}."
    scores = {}
    for res_key, scores_per_res in zip(resolution_keys, scores_list):
        for i, metric_name in enumerate(_AE_RECONSTRUCTION_NAMES):
            if metric_name == 'PSNR':
                # Filter out infinity PSNR values (happens when the image/first frame is black)
                filtered_scores = scores_per_res[:, i][scores_per_res[:, i] != float('+inf')]
                scores[f'{metric_name}_{res_key}_{num_gen}'] = filtered_scores.mean().item() if len(filtered_scores) > 0 else float('inf') # If all are infinity, then ok, return infinity.
            else:
                scores[f'{metric_name}_{res_key}_{num_gen}'] = scores_per_res[:, i].mean().item() # [1]

    # For the highest resolution, save it as a separate key.
    for metric_name in _AE_RECONSTRUCTION_NAMES:
        scores[f'{metric_name}_{num_gen}'] = scores[f'{metric_name}_{resolution_keys[-1]}_{num_gen}']

    return scores

#----------------------------------------------------------------------------
