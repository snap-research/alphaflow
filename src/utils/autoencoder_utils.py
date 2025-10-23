import os
import math
import pickle
from typing import Optional

import torch
import torch.distributions as td
import torch.nn.functional as F

from src.structs import EasyDict
from src.utils import misc, feature_stats, distributed as dist
from src.utils.os_utils import lean_tqdm
from src.data import Data, construct_inputs_from_batch
from src.structs import BasicStats

#----------------------------------------------------------------------------

def vae_latents_forward(
        projector: torch.nn.Module,
        latents: torch.Tensor,
        use_mean_latents: bool=True,
        channel_dim: int=2,
        quantization_precision: Optional[float]=None,
        var_min: float=None,
        prior_stats: torch.Tensor=None,
        vae_rng_seed: Optional[int]=None,
    ) -> tuple[torch.Tensor, EasyDict[str, torch.Tensor]]:

    latents = projector(latents) # [b, f, c, h, w]
    mean, logvar = latents.chunk(2, dim=channel_dim) # [b, f, c, h, w]
    if prior_stats is not None:
        assert latents.shape[1] % prior_stats.shape[0] == 0, f"Number of frames in latents {latents.shape} should be divisible by number of frames in prior_stats {prior_stats.shape}"
        prior_stats = prior_stats.unsqueeze(0).repeat(latents.shape[0], latents.shape[1] // prior_stats.shape[0], 1, 1, 1) # [f, c, h, w]
        prior_mean, prior_logvar = prior_stats.chunk(2, dim=channel_dim) if prior_stats is not None else (None, None) # [b, f, c, h, w], [b, f, c, h, w]
        assert prior_mean.shape == mean.shape, f"prior_mean shape {prior_mean.shape} should be equal to mean shape {mean.shape}"
        assert prior_logvar.shape == logvar.shape, f"prior_logvar shape {prior_logvar.shape} should be equal to logvar shape {logvar.shape}"
    else:
        prior_mean = prior_logvar = None

    if var_min is not None:
        logvar = F.softplus(logvar + math.log(var_min)) - math.log(var_min) # [b, f, c, h, w]; pylint: disable=not-callable
        prior_logvar = F.softplus(prior_logvar + math.log(var_min)) - math.log(var_min) if prior_logvar is not None else None # [b, f, c, h, w]; pylint: disable=not-callable

    if use_mean_latents:
        noise = None
    else:
        noise_generator = None if vae_rng_seed is None else torch.Generator(device=latents.device).manual_seed(vae_rng_seed)
        noise = torch.randn(logvar.shape, generator=noise_generator, device=logvar.device, layout=logvar.layout, dtype=logvar.dtype) * torch.exp(0.5 * logvar) # [b, f, c, h, w]
        if quantization_precision is not None:
            noise = noise + (torch.rand_like(noise) - 0.5) * quantization_precision # [b, f, c, h, w]
    latents = mean if noise is None else (mean + noise) # [b, f, c, h, w]

    kl, kl_raw, entropy, entropy_raw = compute_kl(mean, logvar, prior_mean, prior_logvar, latents, quantization_precision=quantization_precision) # [b]

    return latents, EasyDict(
        kl=kl,
        kl_raw=kl_raw,
        latents_dist_mean=mean,
        latents_dist_logvar=logvar,
        entropy=entropy,
        entropy_raw=entropy_raw,
    )


def compute_kl(mean: torch.Tensor, logvar: torch.Tensor, prior_mean: torch.Tensor, prior_logvar: torch.Tensor, latents: torch.Tensor, quantization_precision: Optional[float]=None):
    entropy_raw = 0.5 * (1 + math.log(2 * math.pi) + logvar) # [b, f, c, h, w]

    if quantization_precision is None:
        # Computing the KL normally (possibly, using the prior).
        if prior_mean is None:
            kl_raw = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp()) # [b, f, c, h, w]
        else:
            kl_raw = 0.5 * (prior_logvar - logvar + (logvar.exp() + (mean - prior_mean).pow(2)) / prior_logvar.exp() - 1)
    else:
        kl_raw = -1.0 * gaussian_log_prob_mass(prior_mean, prior_logvar, latents, bin_size=quantization_precision) # [b, f, c, h, w]

    # Here, we average along dim=channel_dim (channel dimension) instead of summing to make it normalized and comparable between different latent dimensionalities.
    kl = kl_raw.flatten(start_dim=1).mean(dim=1) # [b, f, c, h, w] -> [b, f * c * h * w] -> [b]
    entropy = entropy_raw.flatten(start_dim=1).mean(dim=1) # [b, f, c, h, w] -> [b, f * c * h * w] -> [b]

    return kl, kl_raw, entropy, entropy_raw


def concat_dummy_logvar(x, channel_dim: int=2):
    return torch.cat([x, torch.ones_like(x) * -100], dim=channel_dim)


def gaussian_log_prob_mass(mean, logvar, x, bin_size=1.0, prob_clamp=1e-8):
    """
    Compute log(P) of a "quantized" Normal(`mean`, `logvar`) distribution evaluated at `x`,
    where P = cdf(`x` + 0.5*bin_size) - cdf(`x` - 0.5*bin_size).

    Args:
        mean        (Tensor): mean of the Gaussian
        logvar       (Tensor): logvar (standard deviation) of the Gaussian
        x           (Tensor): the quantized Gaussian is evaluated at `x`
        bin_size    (float):  quantization bin size
        prob_clamp  (float):  when prob < prob_clamp, use approximation to improve numerical stability.
    """
    scale = torch.exp(0.5 * logvar) # [<any>]
    mean, scale, x = misc.to(mean, scale, x, dtype=torch.double)
    log_prob = _safe_log_prob_mass(td.Normal(mean, scale), x, bin_size, prob_clamp)
    return log_prob


@torch.autocast('cuda', enabled=False) # disable mixed precision
def _safe_log_prob_mass(distribution, x, bin_size, prob_clamp):
    # If the probability mass is too small, we use the gaussian density directly to avoid numerical instability.
    prob_mass = distribution.cdf(x + 0.5 * bin_size) - distribution.cdf(x - 0.5 * bin_size)
    log_prob = torch.where(
        prob_mass > prob_clamp,
        torch.log(prob_mass.clamp(min=1e-8)),
        # We can use such an approximation since PDF = dCDF/dx = dCDF/bin_size for bin_size -> 0.
        distribution.log_prob(x) + math.log(bin_size),
    )
    return log_prob

#----------------------------------------------------------------------------

def init_autoencoder_stats(cfg, latents_stats_path: os.PathLike=None, keys: list[str]=None) -> tuple[EasyDict[str, feature_stats.FeatureStats], bool]:
    stats_dict = EasyDict()
    if keys is None or 'latents' in keys: stats_dict.latents = feature_stats.FeatureStats(capture_basic_stats=True)
    if cfg.model.is_vae:
        if keys is None or 'dist_mean' in keys: stats_dict.dist_mean = feature_stats.FeatureStats(capture_basic_stats=True)
        if keys is None or 'dist_std' in keys: stats_dict.dist_std = feature_stats.FeatureStats(capture_basic_stats=True)
        if keys is None or 'kl_raw' in keys: stats_dict.kl_raw = feature_stats.FeatureStats(capture_basic_stats=True)
        if keys is None or 'bpp' in keys: stats_dict.bpp = feature_stats.FeatureStats(capture_basic_stats=True)
    if latents_stats_path is not None and os.path.exists(latents_stats_path):
        assert os.path.isfile(latents_stats_path), f"latents_stats_path {latents_stats_path} should be a file."
        with open(latents_stats_path, 'rb') as f:
            states = EasyDict(pickle.load(f))
        assert set(states.keys()) == set(stats_dict.keys()), f"States keys {set(states.keys())} should be equal to stats_dict keys {set(stats_dict.keys())}"
        for k, s in states.items():
            stats_dict[k] = feature_stats.FeatureStats.load_from_state(s)
        was_loaded = True
    else:
        was_loaded = False
    return stats_dict, was_loaded

@torch.no_grad()
def compute_autoencoder_stats(autoencoder: torch.nn.Module, data: Data, cfg, device, num_samples: int, stats=None, verbose: bool=False, stats_keys=None) -> EasyDict[str, BasicStats]:
    stats, was_loaded = init_autoencoder_stats(cfg, keys=stats_keys) if stats is None else (stats, False)

    if not was_loaded:
        num_batches_to_process = max(num_samples // (data.batch_info.batch_gpu * dist.get_world_size()), 1)
        assert num_batches_to_process > 0, f"Number of batches to process {num_batches_to_process} is not positive. Please increase the number of samples to process."

        for _ in lean_tqdm(range(num_batches_to_process), disable=not (verbose and dist.is_main_process()), desc='Computing autoencoder statistics'):
            batch = next(data.iterator)
            x, cond = construct_inputs_from_batch(batch, use_x_cond=cfg.model.use_x_cond, device=device) # [b, ...], [b, ...]
            # TODO: we should probably use the encode function from the autoencoder?
            sigma = (misc.unwrap_module(autoencoder).cfg.sampling.get('sigma_min', 0.0) + 1e-16) * torch.ones(len(x), 1, 1, 1, 1, device=x.device).float() # [b, 1, 1, 1, 1]
            ignore_labels = cfg.model.get('autoencoder_ignore_labels', False)
            if cfg.model.is_lgm and cfg.model.use_precomputed_latents:
                latents = x # [b, t, c, h, w]
                ctx = None
            else:
                # TODO: that's a dirty hack, but not sure how to make things better. We need to skip the batches which are not our data resolution.
                if (x.shape[1], x.shape[3], x.shape[4]) != tuple(cfg.dataset.resolution):
                    continue

                ctx = autoencoder(x, sigma, cond=cond, return_extra_output=True, encode=True, decode=False, force_label_dropout=ignore_labels)[1] # [b, ...]
                latents = ctx.latents # [b, lt, lc, lh, lw]

            if 'latents' in stats: stats.latents.append_torch(latents.flatten(start_dim=1)) # [b, lt * lc * lh * lw]
            if 'dist_mean' in stats: stats.dist_mean.append_torch(ctx.latents_dist_mean.flatten(start_dim=1)) # [b, lt * lc * lh * lw]
            if 'dist_std' in stats: stats.dist_std.append_torch(ctx.latents_dist_logvar.flatten(start_dim=1).mul(0.5).exp()) # [b, lt * lc * lh * lw]
            if 'kl_raw' in stats: stats.kl_raw.append_torch(ctx.kl_raw.flatten(start_dim=1)) # [b, lt * lc * lh * lw]
            if 'bpp' in stats:
                ctx_q = autoencoder(x, sigma, cond=cond, return_extra_output=True, encode=True, decode=False, quantize=True)[1]
                stats.bpp.append_torch(ctx_q.bpp.unsqueeze(1)) # [b, 1]

    return EasyDict(**{k: v.get_basic_stats() for k, v in stats.items()})

#----------------------------------------------------------------------------
