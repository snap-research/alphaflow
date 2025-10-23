import os
import pickle

import torch
import hydra
from hydra import compose, initialize
from omegaconf import DictConfig
from loguru import logger

from src.structs import EasyDict
from src.metrics import metric_main
from src.utils import distributed as dist
from src.utils.os_utils import disable_trivial_warnings, save_json
from src.data import Data
from src.training.network_utils import load_snapshot
from src.utils.autoencoder_utils import compute_autoencoder_stats, init_autoencoder_stats
from infra.utils import dict_to_hydra_overrides, recursive_instantiate

#----------------------------------------------------------------------------

@hydra.main(config_path="../configs", config_name="evaluate.yaml", version_base='1.2')
def evaluate(cfg: DictConfig):
    disable_trivial_warnings()
    cfg = EasyDict.init_recursively(cfg)

    # Init torch.distributed and torch settings.
    dist.init()
    dist.init_random_state_and_cuda(seed=42, cudnn_benchmark=cfg.cudnn_benchmark, allow_tf32=cfg.allow_tf32)
    device = torch.device('cuda')

    # Loading the info about the generated samples.
    if cfg.gen_dataset is None:
        # Loading the network, and we'll use it to run the evals.
        net, snapshot_path, experiment_cfg = load_snapshot(cfg.ckpt, verbose=dist.get_rank() == 0, device=device)
        if cfg.get('overwrite_path_remote') is not None:
            logger.info('overwriting path_remote', cfg.overwrite_path_remote)
            if len(experiment_cfg.dataset.video_streams) > 0: experiment_cfg.dataset.video_streams[0].path_remote = cfg.overwrite_path_remote
            if len(experiment_cfg.dataset.image_streams) > 0: experiment_cfg.dataset.image_streams[0].path_remote = cfg.overwrite_path_remote
        net = net.to(device).eval()
        gen_dataset = None
        recursive_instantiate(experiment_cfg)
    else:
        # We already produced the samples with the model, and we'll use them to run the evals.
        net = _snapshot = snapshot_path = None
        gen_dataset = Data.init_from_cfg(cfg.gen_dataset).dataset
        experiment_cfg = None

    # Loading the info about the training dataset.
    if 'dataset' not in cfg or cfg.dataset is None:
        if experiment_cfg is None:
            # Initialize the GT dataset kwargs and model config from the provided experiment config.
            assert os.path.isfile(cfg.experiment_cfg.path), f'Invalid cfg.experiment_cfg.path: {cfg.experiment_cfg.path} (we need it for GT dataset kwargs)'
            hydra.core.global_hydra.GlobalHydra.instance().clear()
            with initialize(config_path=os.path.join('../', os.path.dirname(cfg.experiment_cfg.path))):
                overrides = None if len(cfg.experiment_cfg.overrides) == 0 else dict_to_hydra_overrides(cfg.experiment_cfg.overrides)
                experiment_cfg = compose(os.path.basename(cfg.experiment_cfg.path), overrides=overrides if overrides is not None else [])
            recursive_instantiate(experiment_cfg)
        gt_dataset_cfg = EasyDict.init_recursively(experiment_cfg.dataset)
    else:
        gt_dataset_cfg = EasyDict.init_recursively(cfg.dataset)

    # TODO: this patching is too hacky, we should do something better.
    experiment_cfg = EasyDict.init_recursively({'dataset': {}} if experiment_cfg is None else experiment_cfg)

    data: Data = Data.init_from_cfg(gt_dataset_cfg)

    if any(m.startswith('lat_noised_reconstruction') for m in cfg.metrics):
        latents_stats_path = os.path.join(cfg.env.latents_stats_dir, f'{experiment_cfg.experiment_id_str}-{os.path.basename(snapshot_path)}-n128.pkl')
        stats, was_loaded = init_autoencoder_stats(experiment_cfg, latents_stats_path=latents_stats_path, keys=['latents'])
        experiment_cfg.dataset.predownload = 0
        if was_loaded:
            latents_std = stats.latents.get_basic_stats().std # [t * c * h * w]
        else:
            latents_std = compute_autoencoder_stats(net, data, experiment_cfg, device, num_samples=128, stats=stats).latents.std # [t * c * h * w]
            # Save the stats.
            with open(latents_stats_path, 'wb') as f:
                pickle.dump(stats, f)
        latents_std = torch.from_numpy(latents_std).float().to(device).view(-1, net.model.latent_channels, *net.model.latent_resolution[1:]) # [t, c, h, w]
    else:
        latents_std = None

    # Validate arguments.
    assert len(cfg.metrics) > 0, f'cfg.metrics must contain at least one value, but got {cfg.metrics}'
    assert all(metric_main.is_valid_metric(m) for m in cfg.metrics), '\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics())
    assert dist.get_world_size() >= 1, f'--gpus must be at least 1, but have {dist.get_world_size()}'
    if gen_dataset is not None:
        if "streams" in cfg.gen_dataset and len(cfg.gen_dataset.streams) > 1:
            gen_path = cfg.gen_dataset.streams[0].src
        else:
            gen_path = cfg.gen_dataset.src
    else:
            gen_path = snapshot_path

    # Calculate each metric.
    results = {}
    for metric in cfg.metrics:
        if cfg.verbose:
            dist.loginfo0(f'Calculating {metric} for {gen_path}...')
        result_dict = metric_main.compute_metric(
            metric,
            net=net,
            dataset=data.dataset_eval,
            model_kwargs=cfg.model_kwargs,
            rank=dist.get_rank(),
            device=torch.device('cuda', dist.get_rank()),
            verbose=cfg.verbose,
            gen_dataset=gen_dataset,
            batch_gen=cfg.batch_size,
            sampling_cfg=cfg.get('sampling'),
            dataset_stats_dir=cfg.env.dataset_stats_dir,
            detector_batch_gpu=cfg.detector_batch_gpu,
            save_partial_stats_freq=cfg.save_partial_stats_freq,
        )
        results[metric] = result_dict
        if dist.is_main_process():
            metric_main.report_metric(result_dict, run_dir=None, snapshot_path=snapshot_path, save_result=False)
        if cfg.verbose:
            dist.print0()

    if cfg.save_path is not None and dist.is_main_process():
        os.makedirs(os.path.dirname(cfg.save_path), exist_ok=True)
        save_json(results, cfg.save_path)

    # Done.
    if cfg.verbose:
        dist.loginfo0('Exiting...')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    evaluate() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
