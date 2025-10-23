from typing import Sequence, Callable
from omegaconf import DictConfig

import numpy as np
import torch
import einops

from src.structs import TensorLike
from src.structs import EasyDict, TensorGroup
from src.utils.os_utils import lean_tqdm
from src.utils import misc

#----------------------------------------------------------------------------

@torch.no_grad()
@misc.profiled_function
def sample(
        net: torch.nn.Module,
        cfg: DictConfig,
        seeds: list,
        cond: Sequence[TensorGroup | None],
        verbose: bool=False,
        device: str=torch.device('cuda'),
        save_traj_len: int=1,
        **sample_batch_kwargs,
    ) -> TensorLike:
    """
    Samples videos from the model. Returns them in the [0, 255] range with uint8 type.
    It takes a list of lists as seeds/conditionings to avoid writing batch-wise inference loops in all the calling functions.
    """
    assert len(seeds) > 0, f"Empty seeds list: {len(seeds)}"
    assert len(seeds) == len(cond), f"Seeds and cond lists have different lengths: {len(seeds)} != {len(cond)}. You should provide empty cond if cond are not used."
    assert all(c is None for c in cond) or [len(s) for s in seeds] == [len(s) for s in cond], f"Seeds and cond lists have different batch sizes: {[len(s) for s in seeds]} != {[len(s) for s in cond]}"

    trajectories = []
    is_training_mode = net.training
    net.eval()

    for subbatch_idx, batch_seeds in lean_tqdm(enumerate(seeds), unit='batch', disable=not verbose, total=len(seeds)):
        if len(batch_seeds) == 0:
            continue

        # Preparing conditioning.
        cond_cur = cond[subbatch_idx] # [b, *labels_shape] or None
        cond_cur = None if cond_cur is None else cond_cur.to(device) # [b, *labels_shape] or None
        samples = sample_batch_diffusion(cfg, net, batch_seeds, device=device, cond=cond_cur, save_traj_len=save_traj_len, **sample_batch_kwargs) # [traj_len, b, t, c, h, w]
        samples = misc.rearrange(samples, 'l b t c h w -> b l t h w c') # [b, traj_len, t, h, w, c]
        if isinstance(samples, TensorGroup):
            samples.video = misc.maybe_rescale_fp32_to_uint8(samples.video) # [b, traj_len, t, h, w, c]
            if 'audio' in samples: samples.audio = einops.rearrange(samples.audio, 'b l t 1 1 c -> b l c t') # [b, l, ca, ta]
            trajectories.extend([samples[[i]] for i in range(len(samples))]) # [cur_num_seeds, traj_size, t, h, w, c]
        else:
            samples = misc.maybe_rescale_fp32_to_uint8(samples) # [b, traj_len, t, h, w, c]
            trajectories.extend(samples) # [cur_num_seeds, traj_size, t, h, w, c]

    trajectories = misc.stack(trajectories, dim=0) # [num_seeds, traj_size, t, h, w, c]
    trajectories = trajectories.squeeze(1) if save_traj_len == 1 else trajectories # [num_seeds, traj_size | null, t, h, w, c]
    net.train(is_training_mode)

    return trajectories # [num_seeds, traj_size | null, t, h, w, c]

#----------------------------------------------------------------------------

@misc.profiled_function
def sample_batch_diffusion(
        cfg,
        net,
        batch_seeds,
        device,
        save_traj_len: int=1,
        cond: torch.Tensor=None,
        **sampler_kwargs,
    ) -> torch.Tensor:
    net_bare = misc.unwrap_module(net) # For accessing methods and properties.
    rnd = misc.StackedRandomGenerator(device, batch_seeds) # (batch_size, <SRG>)
    noise = rnd.randn([len(batch_seeds), *net_bare.input_shape], device=device) # [b, t, c, h, w]

    if cfg.real_init:
        inits = cond.x_cond.to(device) # [b, t_l, c, h_l, w_l]
        if net_bare.cfg.is_lgm:
            inits = net_bare.encode(inits, cond) # [b, t_lat, c, h_lat, w_lat]
    else:
        inits = misc.zeros_like(noise) # [b, t, c, h, w] or {k: [b, t, c, h, w]}

    videos: torch.Tensor = run_reverse_diffusion(
        cfg, net, noise, randn_like=rnd.randn_like, inits=inits,
        use_sc_latents=net_bare.cfg.get('self_cond_probability') > 0.0,
        save_traj_len=save_traj_len, cond=cond, **sampler_kwargs,
    ) # [traj_len, batch_size, t_l, c, h_l, w_l]

    if net_bare.cfg.is_lgm:
        ae_sampling_kwargs = dict(cfg=EasyDict(**{**net_bare.autoencoder.cfg.sampling, **dict(real_init=False)}), net=net_bare.autoencoder, batch_seeds=batch_seeds, device=device, encode=False)
        videos = torch.stack([sample_batch_diffusion(**ae_sampling_kwargs, cond=cond, latents=net_bare.denormalize_latents(x.to(device))).squeeze(0) for x in videos], dim=0) # [traj_len, b, t, c, h, w]

    if net_bare.cfg.is_autoencoder and save_traj_len > 1 and cond.get('x_cond') is not None:
        # Also save the GT videos to visualize the reconstruction quality.
        diffs = (videos[-1] - cond.x_cond.to(videos.device)).abs() # [b, t_l, c, h_l, w_l]
        to_concat = ([] if cfg.real_init else [cond.x_cond.to(videos.device).unsqueeze(0)]) + [videos] + [diffs.unsqueeze(0)] # (traj_len + 1|2, [b, t_l, c, h_l, w_l])
        videos = torch.cat(to_concat, dim=0) # [traj_len + 1, b, t_l, c, h_l, w_l]

    return videos # [traj_len, b, t, c, h, w]

#----------------------------------------------------------------------------
# Key diffusion sampling functions.

@misc.profiled_function
def run_reverse_diffusion(sampling_cfg: DictConfig, net, noise: TensorLike, inits: TensorLike, save_traj_len: int=1, save_traj_step_idx=None, dtype=torch.float64, **step_kwargs) -> TensorLike:
    batch_size, device = len(noise), noise.device
    t_steps = compute_t_steps(sampling_cfg, stage_idx=0, device=device) # [num_steps + 1]
    t_steps = einops.repeat(t_steps, 'n -> n batch_size 1 1 1 1', batch_size=batch_size).clone() # [num_steps + 1, batch_size, 1, 1, 1, 1]
    t_steps = TensorGroup({k: t_steps for k in noise}) if isinstance(noise, TensorGroup) else t_steps # <k, [num_steps + 1, batch_size, 1, 1, 1, 1]>
    if sampling_cfg.name == 'recflow' and sampling_cfg.time_shifting != 1.0:
        t_steps = 1 - (1 - t_steps) / (sampling_cfg.time_shifting + (1 - sampling_cfg.time_shifting) * (1 - t_steps)) # [num_steps + 1, batch_size, 1, 1, 1, 1]
    save_traj_len = min(save_traj_len, len(t_steps))
    x_next = init_traj_start_latent(sampling_cfg, inits, noise.to(dtype), t_steps[0]).float() # [b, t, c, h, w]
    selected_traj_idx = [len(t_steps) - 1] if save_traj_len == 1 else get_save_traj_idx(save_traj_step_idx, save_traj_len, len(t_steps)) # [save_traj_len]
    trajectory = [x_next.cpu()] if 0 in selected_traj_idx else [] # (1, [batch_size, t, c, h, w])
    reverse_step_fn = reverse_recflow_step if sampling_cfg.name == 'recflow' else reverse_edm_step

    # Main sampling loop.
    sc_latents = None
    for i in range(len(t_steps) - 1): # 0, ..., N-1
        x_next, x_denoised, sc_latents = reverse_step_fn(
            net=net, sampling_cfg=sampling_cfg, x_cur=x_next, t_cur=t_steps[i], t_next=t_steps[i + 1], sc_latents=sc_latents,
            step_idx=i, num_steps=len(t_steps) - 1, dtype=dtype, **step_kwargs) # [b, t, c, h, w], [b, t, c, h, w]
        if (i + 1) in selected_traj_idx: # pylint: disable=superfluous-parens
            trajectory.append(x_denoised.float().cpu())

    trajectory = misc.stack(trajectory) # [traj_len, b, t, c, h, w]
    assert len(trajectory) == save_traj_len == len(selected_traj_idx), f"Incorrect output shape: {trajectory.shape}, {save_traj_len}, {selected_traj_idx}"
    return trajectory

@misc.profiled_function
def reverse_edm_step(
        net: torch.nn.Module,
        sampling_cfg: DictConfig,
        x_cur: TensorLike,
        t_cur: torch.Tensor,
        t_next: torch.Tensor,
        randn_like: Callable=None,
        dtype=None,
        step_idx: int=None,
        num_steps: int=None,
        sc_latents: torch.Tensor=None,
        use_sc_latents: bool=False,
        **model_kwargs,
    ) -> tuple[TensorLike, TensorLike, TensorLike]:
    net_bare = misc.unwrap_module(net)
    x_noised, t_hat = add_extra_noise(sampling_cfg, num_steps, x_cur, t_cur, randn_like)

    def denoise(x_noised, t, guidance: float):
        x_denoised, extra_output = net(x_noised, t, sc_latents=sc_latents, return_extra_output=True, **model_kwargs)
        x_denoised = x_denoised.to(dtype) # [b, t, c, h, w]
        if guidance != 1:
            x_denoised_uncond = net(x_noised, t, sc_latents=sc_latents, return_extra_output=True, force_label_dropout=True, **model_kwargs)[0].to(dtype)
            x_denoised = x_denoised_uncond + guidance * (x_denoised - x_denoised_uncond) # [b, t, c, h, w]
        return x_denoised, extra_output

    # Euler step.
    x_denoised, extra_output = denoise(x_noised, t_hat, sampling_cfg.get('cfg', {}).get('scale', 1.0)) # [b, t, c, h, w], <any>
    sc_latents = extra_output.sc_latents if use_sc_latents else None # [b, lat_count, lat_dim]
    d_cur = (x_noised - x_denoised) / t_hat # [b, t, c, h, w]
    x_next = x_noised + (t_next - t_hat) * d_cur # [b, t, c, h, w]

    # Apply 2nd order correction.
    if sampling_cfg.second_order and step_idx < num_steps - 1:
        x_denoised, extra_output = denoise(x_next, t_next, sampling_cfg.cfg.scale) # [b, t, c, h, w], <any>
        sc_latents = extra_output.sc_latents if use_sc_latents else None # [b, lat_count, lat_dim]
        d_prime = (x_next - x_denoised) / t_next # [b, t, c, h, w]
        x_next = x_noised + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime) # [b, t, c, h, w]

    return x_next, x_denoised, sc_latents

@misc.profiled_function
def reverse_recflow_step(
        net: torch.nn.Module,
        sampling_cfg: DictConfig,
        x_cur: TensorLike,
        t_cur: TensorLike,
        t_next: TensorLike,
        randn_like: Callable=None,
        dtype=None,
        step_idx: int=None,
        num_steps: int=None,
        sc_latents: torch.Tensor=None,
        use_sc_latents: bool=False,
        **model_kwargs,
    ) -> tuple[TensorLike, TensorLike]:
    _ = step_idx, randn_like, t_next, num_steps # For compatibility with reverse_edm_step.

    delta_t = t_cur - t_next # [b, 1, 1, 1, 1]
    sigma_next = torch.zeros_like(t_next) if sampling_cfg.enable_consistency_sampling else t_next
    model_kwargs.update({"sigma_next": sigma_next} if sampling_cfg.enable_trajectory_sampling else {})
    velocity_cur, extra_output = net(x_cur, sigma=t_cur, sc_latents=sc_latents, return_extra_output=True, **model_kwargs) # [b, t, c, h, w], <any>

    # Applying CFG (supports batch-wise and spatio-temporal difference).
    t_cur_item = t_cur[0].item() if isinstance(t_cur, torch.Tensor) else next(iter(t_cur[0].values())).item() # [1]
    cur_cfg_scale = sampling_cfg.cfg.scale if sampling_cfg.cfg.t_min <= t_cur_item <= sampling_cfg.cfg.t_max else 1.0 # [1]
    if cur_cfg_scale != 1.0:
        velocity_cur_uncond, _ = net(x_cur, t_cur, sc_latents=sc_latents, return_extra_output=True, force_label_dropout=True, **model_kwargs) # [b, t, c, h, w], <any>
        velocity_cur = velocity_cur_uncond + cur_cfg_scale * (velocity_cur - velocity_cur_uncond) # [b, t, c, h, w]

    velocity_cur = velocity_cur.to(dtype) # [b, t, c, h, w]
    sc_latents = extra_output.sc_latents if use_sc_latents else None # [b, lat_count, lat_dim]

    x_denoised = x_cur - t_cur * velocity_cur # [b, t, c, h, w]
    if not sampling_cfg.enable_consistency_sampling:
        # Euler step in reverse time
        x_next = x_cur - delta_t * velocity_cur # [b, t, c, h, w]
    else:
        # consistency sampling steps in reverse time
        x_next = (1 - t_next) * x_denoised + t_next * randn_like(x_denoised) # [b, t, c, h, w]


    return x_next, x_denoised, sc_latents

#----------------------------------------------------------------------------
# Diffusion inference helper utils.

@misc.profiled_function
def compute_t_steps(sampling: DictConfig, stage_idx: int=0, device='cpu', dtype=torch.float64):
    if sampling.custom_t_steps is not None:
        t_steps = torch.as_tensor(sampling.custom_t_steps, device=device).double() # [num_steps + 1]
        num_steps = len(t_steps) - 1 # [1]
        return t_steps

    num_steps = sampling.num_steps if stage_idx == 0 else max(int(sampling.num_steps * sampling.num_steps_decay_rate ** stage_idx), sampling.num_steps_min) # [1]
    assert num_steps > 0, f"num_steps should be positive, but got {num_steps}."

    if sampling.name == 'recflow':
        # `step_numbers` have the values: [N, ..., 1, 0].
        step_numbers = torch.arange(num_steps, 0, step=-1, dtype=dtype, device=device) # [num_steps]
        delta_t = (1 - sampling.eps) / num_steps # [1]
        # `t_steps` have the values: [1 - eps, ..., 2 * delta_t, delta_t, 0]
        t_steps = step_numbers * delta_t # [num_steps]
    else:
        if num_steps == 1:
            t_steps = torch.ones(1, dtype=dtype, device=device) * sampling.sigma_max # [1]
        else:
            step_indices = torch.arange(num_steps, dtype=dtype, device=device) # [num_steps]
            t_steps = (sampling.sigma_max ** (1 / sampling.rho) + step_indices / (num_steps - 1) * (sampling.sigma_min ** (1 / sampling.rho) - sampling.sigma_max ** (1 / sampling.rho))) ** sampling.rho # [num_steps]

    # Adding 0 as the last step, but it shouldn't be used directly (only for computing t_hat and stuff).
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # [num_steps + 1]. t_N = 0

    return t_steps

def add_extra_noise(sampling_cfg, num_steps: int, x_cur: torch.Tensor, t_cur: torch.Tensor, randn_like, force_sigma: float=None) -> tuple[torch.Tensor, torch.Tensor]:
    if force_sigma is not None:
        t_cur = t_cur.clone().fill_(force_sigma) # [b, 1, 1, 1, 1]
    # Increase noise temporarily.
    gamma = min(sampling_cfg.S_churn / num_steps, np.sqrt(2) - 1) if sampling_cfg.S_min <= (t_cur[0] if t_cur.ndim == 5 else t_cur[0, -1]).item() <= sampling_cfg.S_max else 0.0 # [1]
    t_hat = t_cur + gamma * t_cur # [b, 1, 1, 1, 1]
    t_extra = (t_hat ** 2 - t_cur ** 2).sqrt() * sampling_cfg.S_noise # [b, 1, 1, 1, 1]
    x_noised = x_cur + t_extra * randn_like(x_cur) # [b, t, c, h, w]

    return x_noised, t_hat

def init_traj_start_latent(sampling_cfg: DictConfig, videos_init: torch.Tensor, noise: torch.Tensor, t: torch.Tensor):
    """For various diffision parametrizations, we use different ways to start the trajectory"""
    assert videos_init.shape == noise.shape, f"videos_init and noise shapes do not match: {videos_init.shape} != {noise.shape}"
    assert misc.is_strictly_broadcastable(videos_init, t), f"videos_init and t shapes do not correspond: {videos_init.shape} vs {t.shape}"
    if sampling_cfg.name in ['edm', 'autoencoder']:
        return videos_init + t * noise # [b, t, c, h, w]
    elif sampling_cfg.name == 'recflow':
        return (1.0 - t) * videos_init + sampling_cfg.sigma_noise * t * noise # [b, t, c, h, w]
    else:
        raise NotImplementedError(f"Unknown precond: {sampling_cfg.name}")

def get_save_traj_idx(save_traj_step_idx, save_traj_len, num_steps: int) -> np.ndarray:
    if save_traj_step_idx is None:
        return np.floor(np.linspace(0, num_steps - 1, save_traj_len)).astype(int) # [traj_len]
    else:
        assert min(save_traj_step_idx) >= 0 and max(save_traj_step_idx) < num_steps, f"save_traj_step_idx should be in the range [0, {num_steps - 1}], but got {save_traj_step_idx}"
        return np.array(save_traj_step_idx) # [traj_len]

#----------------------------------------------------------------------------
