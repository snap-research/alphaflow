from typing import Optional, Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import einops
import lpips
import math
from omegaconf import DictConfig
from torch.autograd.functional import jvp as jvp
from beartype import beartype


from src.structs import EasyDict, TensorGroup
from src.utils import misc
from src.utils.training_utils import sample_frames_masks, cut_dct2d_high_freqs, reg_dc_dct2d_high_freqs, compute_annealed_weight, compute_scheduled_weight
from src.training.perceptual_loss import PerceptualPyramidLoss
from src.structs import TokenType, LossPhase, TensorLike, BaseLoss
from src.training.network_utils import load_snapshot

#----------------------------------------------------------------------------
# Some constants.

SOBEL_FILTER = [
    [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
    [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
] # [2, 3, 3]

#----------------------------------------------------------------------------

class DiffusionLoss(BaseLoss):
    def __init__(self, cfg: EasyDict):
        super().__init__()
        self.cfg = cfg
        self.teacher = None

        if self.cfg.use_single_step_denoising:
            self._init_lpips_models()

    def _get_sigma_shape(self, input_shape: torch.Size) -> torch.Size:
        if self.cfg.per_pixel_noise_level:
            return torch.Size(input_shape[:2]) + torch.Size([1]) + torch.Size(input_shape[3:]) # [5]. Shape is [b, t, 1, h, w]
        else:
            return torch.Size(input_shape[:1]) + torch.Size([1] * (len(input_shape) - 1)) # [5]. Shape is [b, 1, 1, 1, 1]

    def _init_lpips_models(self):
        self.lpips_pyr = PerceptualPyramidLoss(
            scales=self.cfg.perceptual_loss.scales,
            loss_weights=self.cfg.perceptual_loss.weights,
            replace_maxpool_with_avgpool=self.cfg.perceptual_loss.replace_maxpool_with_avgpool,
            downsample_to_native=self.cfg.perceptual_loss.downsample_to_native,
        ).requires_grad_(False) if self.cfg.perceptual_loss.weight > 0 else None
        self.lpips_alex = lpips.LPIPS(net='alex').requires_grad_(False) if self.cfg.perceptual_loss_alex.weight > 0 else None
        self.lpips_vgg = lpips.LPIPS(net='vgg').requires_grad_(False) if self.cfg.perceptual_loss_vgg.weight > 0 else None
        self.lpips_squeeze = lpips.LPIPS(net='squeeze').requires_grad_(False) if self.cfg.perceptual_loss_squeeze.weight > 0 else None

    def sample_sigma(self, batch_size, device, cfg=None) -> torch.Tensor:
        raise NotImplementedError

    def compute_loss_weight(self, sigma: torch.Tensor, logvar: torch.Tensor=None) -> torch.Tensor:
        raise NotImplementedError

    def apply_noise(self, videos: torch.Tensor, noise_scaled: torch.Tensor, sigma: torch.Tensor, net: torch.nn.Module) -> torch.Tensor: # pylint: disable=unused-argument
        raise NotImplementedError

    def compute_x_denoised(self, net_output: torch.Tensor, videos_noised: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """x_denoised for single-step trajectory solving"""
        raise NotImplementedError

    @torch.no_grad()
    def _maybe_compute_sc_latents(self, net, videos_aug_noised, sigma, cond, **kwargs) -> torch.Tensor:
        if self.cfg.model.self_cond_probability > 0.0:
            sc_latents = net(videos_aug_noised, sigma, cond, return_extra_output=True, **kwargs)[1]['sc_latents'] # [b, lt, c, lh, lw] or [b, num_latents, lat_dim]
        else:
            sc_latents = None
        return sc_latents

    @torch.no_grad()
    def _maybe_encode(self, net, x: TensorLike, cond) -> TensorLike:
        if misc.unwrap_module(net).cfg.is_lgm:
            return misc.unwrap_module(net).encode(x, cond=cond, only_normalize=self.cfg.model.use_precomputed_latents)
        else:
            return x

    def compute_kl_loss(self, loss_total: torch.Tensor, ctx: EasyDict, cur_step: int=None, loss_weight: torch.Tensor=None) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.cfg.model.is_vae:
            loss_kl = ctx['kl'] # [b]
            loss_kl_diffusion_weighted = (loss_weight * loss_kl) if (loss_weight is not None and self.cfg.apply_denoising_loss_weight_to_reg) else loss_kl # [b]
            raw_kl_loss_weight: float = self.get_cur_kl_loss_weight(cur_step) # [1]
            if self.cfg.kl_weight_relative:
                loss_kl_weighted = reweigh_supp_loss(loss_total, loss_kl_diffusion_weighted, raw_kl_loss_weight) # [b]
            else:
                loss_kl_weighted = raw_kl_loss_weight * loss_kl_diffusion_weighted # [b]
        else:
            loss_kl = loss_kl_weighted = None

        return EasyDict(kl=loss_kl, kl_weighted=loss_kl_weighted)

    def maybe_apply_reg(self, loss_total: torch.Tensor, ctx: EasyDict, cur_step: int=None, **kwargs) -> tuple[torch.Tensor, EasyDict[str, Optional[torch.Tensor]]]:
        kl_losses = self.compute_kl_loss(loss_total, ctx, cur_step, **kwargs) # <str, [b]>
        if kl_losses.kl_weighted is not None:
            loss_total = loss_total + kl_losses.kl_weighted # [b]
        reg_losses = EasyDict(**kl_losses) # <str, [b]>
        return loss_total, reg_losses

    def compute_rec_losses(self, videos_pred: torch.Tensor, targets: torch.Tensor, frames_mask: torch.Tensor=None, loss_weights_overrides: dict=None) -> EasyDict[str, torch.Tensor]:
        loss_weights = EasyDict(
            lpips_pyr=self.cfg.perceptual_loss.weight,
            lpips_alex=self.cfg.perceptual_loss_alex.weight,
            lpips_vgg=self.cfg.perceptual_loss_vgg.weight,
            lpips_squeeze=self.cfg.perceptual_loss_squeeze.weight,
            mse=self.cfg.mse_loss.weight,
            mae=self.cfg.mae_loss.weight,
            pseudo_huber=self.cfg.pseudo_huber_loss.weight,
            img_grad=self.cfg.img_grad_loss.weight,
            freq2d=self.cfg.freq2d_loss.weight,
            freq3d=self.cfg.freq3d_loss.weight,
            random_conv_l2=self.cfg.random_conv_l2_loss.weight,
            video_random_conv_l2=self.cfg.video_random_conv_l2_loss.weight,
        )
        loss_weights = EasyDict(**{**loss_weights, **loss_weights_overrides}) if loss_weights_overrides is not None else loss_weights
        loss_fns = EasyDict(
            lpips_pyr = self.lpips_pyr, # [b, t, 1, 1, 1]
            lpips_alex = lambda x_rec, x_gt: compute_video_lpips(x_rec, x_gt, self.lpips_alex, self.cfg.perceptual_loss_alex), # [b, t, 1, 1, 1]
            lpips_vgg = lambda x_rec, x_gt: compute_video_lpips(x_rec, x_gt, self.lpips_vgg, self.cfg.perceptual_loss_vgg), # [b, t, 1, 1, 1]
            lpips_squeeze = lambda x_rec, x_gt: compute_video_lpips(x_rec, x_gt, self.lpips_squeeze, self.cfg.perceptual_loss_squeeze), # [b, t, 1, 1, 1]
            mse = lambda x_rec, x_gt: ((x_rec - x_gt) ** 2), # [b, t, c, h, w]
            mae = lambda x_rec, x_gt: (x_rec - x_gt).abs(), # [b, t, c, h, w]
            pseudo_huber = lambda x_rec, x_gt: ((x_rec - x_gt) ** 2 + self.cfg.pseudo_huber_loss.breadth_coef ** 2).sqrt() - self.cfg.pseudo_huber_loss.breadth_coef, # [b, t, c, h, w]
            img_grad = compute_img_grad_loss_per_frame, # [b, t, 1, 1, 1]
            freq2d = compute_video_freq2d_loss, # [b, t, 1, 1, 1]
            freq3d = compute_video_freq3d_loss, # [b, t, 1, 1, 1]
            random_conv_l2 = lambda x_rec, x_gt: compute_framewise_video_random_conv_l2(x_rec, x_gt, self.cfg.random_conv_l2_loss), # [b, t, 1, 1, 1]
            video_random_conv_l2 = lambda x_rec, x_gt: compute_video_random_conv_l2(x_rec, x_gt, self.cfg.video_random_conv_l2_loss), # [b, t, 1, 1, 1]
        )
        losses_all = EasyDict(**{k: loss_fn(videos_pred, targets) if loss_weights[k] > 0 else None for k, loss_fn in loss_fns.items()})
        losses_all = misc.filter_nones(losses_all)
        losses_all_filtered = EasyDict(**{k: maybe_filter_loss_by_mask(l, frames_mask) for k, l in losses_all.items()}) # [b, t, c, h, w]
        losses_all_filtered_agg = EasyDict(**{k: l.reshape(len(l), -1).mean(dim=1, keepdim=True) for k, l in losses_all_filtered.items()}) # [b]
        losses_all_filtered_agg_weighted = EasyDict(**{k: (l * loss_weights[k]) for k, l in losses_all_filtered_agg.items()}) # [b]
        loss_total_rec = sum(l for l in losses_all_filtered_agg_weighted.values()) # [b]
        return EasyDict(rec=loss_total_rec, **losses_all_filtered_agg) # Returning unweighted losses for logging purposes.

    def get_cur_kl_loss_weight(self, cur_step: int) -> float:
        if self.cfg.model.is_vae:
            if len(self.cfg.kl_weight_schedule) > 0:
                assert self.cfg.kl_weight == 0.0, f"Expected kl_weight to be 0.0, but got {self.cfg.kl_weight} instead."
                cur_kl_weight: float = compute_scheduled_weight(cur_step, self.cfg.kl_weight_schedule)
            elif self.cfg.kl_weight_anneal_steps is not None:
                cur_kl_weight = compute_annealed_weight(cur_step, start_weight=0.0, end_weight=self.cfg.kl_weight, annealing_steps=self.cfg.kl_weight_anneal_steps)
            else:
                cur_kl_weight = self.cfg.kl_weight
        else:
            cur_kl_weight = 0.0
        return cur_kl_weight

    # This functon shouldn't be overridden. Instead, override the methods.
    def forward(self, net, x: TensorLike, cond: TensorGroup, augment_pipe=None, phase=LossPhase.Gen, cur_step=None, force_sigma_val: Optional[float]=None) -> EasyDict[str, torch.Tensor]:
        _ = phase # Unused.
        assert isinstance(x, TensorGroup) or x.ndim == 5, f"Expected tensor x to be videos and have 5 dimensions: [b, t, c, h, w], got {x.shape} instead."
        x, augment_labels = maybe_augment_videos(x, augment_pipe) # [b, t, c, h, w], [b, augment_dim] or None
        x = self._maybe_encode(net, x, cond) # [b, t, c, h, w]
        sigma = self.sample_sigma(x.shape, x.device) # [b, ...]
        if force_sigma_val is not None:
            sigma.fill_(force_sigma_val)
        noise_unscaled = misc.randn_like(x) # [b, t, c, h, w]
        noise_scaled = noise_unscaled * sigma # [b, t, c, h, w]
        videos_aug_noised = self.apply_noise(x, noise_scaled, sigma, net) # [b, t, c, h, w]
        sc_latents = self._maybe_compute_sc_latents(net, videos_aug_noised, sigma, cond, augment_labels=augment_labels) # [b, num_latents, lat_dim] or None
        net_output, ctx = net(videos_aug_noised, sigma, cond, augment_labels=augment_labels, sc_latents=sc_latents, return_extra_output=True) # [b, t, c, h, w]
        net_output = TensorGroup(net_output) if isinstance(net_output, dict) else net_output # [b, t, c, h, w]
        targets = self.compute_targets(x, noise_unscaled, sigma) # [b, t, c, h, w]
        if isinstance(net_output, TensorGroup) and 'audio' in net_output:
            targets.audio = targets.audio[:, :net_output.audio.shape[1], :] # [b, t_a, c_a, 1, 1]
        loss_diffusion = (net_output - targets) ** 2 # [b, t, c, h, w]
        loss_rec = self.cfg.denoising_loss_weight * loss_diffusion # [b, t, c, h, w]
        if self.cfg.use_single_step_denoising:
            videos_denoised = self.compute_x_denoised(net_output, videos_aug_noised, sigma) # [b, t, c, h, w]
            single_step_rec_losses = self.compute_rec_losses(videos_denoised, x) # [b]
            single_step_rec_losses.rec_single_step = single_step_rec_losses.pop('rec') # [b]
            loss_rec = loss_rec + single_step_rec_losses.rec_single_step # [b, t, c, h, w]
        else:
            single_step_rec_losses = EasyDict({})
        loss_weight = self.compute_loss_weight(sigma, logvar=ctx.get('logvar')) # [b, ...]
        loss_total = loss_weight * loss_rec + (ctx['logvar'] if ctx.get('logvar') is not None else 0.0) # [b, t, c, h, w]
        loss_total, reg_losses_dict = self.maybe_apply_reg(loss_total, ctx, cur_step=cur_step, loss_weight=loss_weight) # [b], <str, [b]>
        loss_dict = misc.filter_nones(_maybe_convert_tensor_group(EasyDict(
            total=loss_total, # [b]
            logvar_uncertainty=ctx.get('logvar'), # [b, tl, c, hl, wl]
            rec=loss_rec * loss_weight, # [b]
            rec_unweighted=loss_rec, # [b]
            diffusion=loss_weight * loss_diffusion, # [b]
            diffusion_unwheighted=loss_diffusion, # [b]
            **reg_losses_dict, **single_step_rec_losses)))
        return loss_dict

#----------------------------------------------------------------------------

class EDMLoss(DiffusionLoss):
    def sample_sigma(self, shape: torch.Size, device, cfg=None) -> torch.Tensor:
        cfg = self.cfg if cfg is None else cfg
        rnd_normal = torch.randn(self._get_sigma_shape(shape), device=device) # [b, ...]
        return (rnd_normal * cfg.P_std + self.cfg.P_mean).exp() # [b, ...]

    def apply_noise(self, videos: torch.Tensor, noise_scaled: torch.Tensor, sigma: torch.Tensor, net: torch.nn.Module) -> torch.Tensor: # pylint: disable=unused-argument
        return videos + noise_scaled

    def compute_targets(self, videos_gt: torch.Tensor, noise_unscaled: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor: # pylint: disable=unused-argument
        return videos_gt

    def compute_loss_weight(self, sigma: torch.Tensor, logvar: torch.Tensor=None) -> torch.Tensor:
        if self.cfg.model.sampling.sigma_max == 0.0:
            return misc.ones_like(sigma)
        return (sigma ** 2 + self.cfg.model.sigma_data ** 2) / (sigma * self.cfg.model.sigma_data) ** 2 / (logvar.exp() if logvar is not None else 1.0)

#----------------------------------------------------------------------------

class RecFlowLoss(DiffusionLoss):
    def sample_sigma(self, shape: torch.Size, device, cfg=None) -> torch.Tensor:
        cfg = self.cfg.time_sampling if cfg is None else cfg
        shape = self._get_sigma_shape(shape)
        if cfg.custom_t_steps is None:
            return sample_logit_norm_time(cfg, shape, device=device) # [b, ...]
        else:
            return sample_custom_t_steps(cfg.custom_t_steps, shape, device=device) # [b, ...]

    def compute_loss_weight(self, sigma: TensorLike, logvar: TensorLike=None) -> TensorLike:
        return misc.ones_like(sigma) / (logvar.exp() if logvar is not None else 1.0) # [b, 1, 1, 1, 1]

    def apply_noise(self, videos: TensorLike, noise_scaled: TensorLike, sigma: TensorLike, net: torch.nn.Module) -> TensorLike:
        # Noise has already been scaled by sigma, the only extra scaling for it is sigma_noise.
        return (1 - sigma) * videos + self.cfg.model.sampling.sigma_noise * noise_scaled # [b, t, c, h, w]

    def compute_targets(self, videos_gt: TensorLike, noise_unscaled: TensorLike, sigma: TensorLike) -> TensorLike:
        _ = sigma # Unused.
        return noise_unscaled * self.cfg.model.sampling.sigma_noise - videos_gt # [b, t, c, h, w]

    def compute_x_denoised(self, net_output: TensorLike, videos_noised: TensorLike, sigma: TensorLike) -> TensorLike:
        # Sometimes we want to do kinda one-step diffusion, where the target is the clean video.
        # For rectified flows:
        #     - net_output is the velocity: v = noise - x_0 (independent of the time step) of going from x_0 to x_1.
        #     - sigma is time t (the distance betwen noise = z_0 and z_t)
        #     - the starting point z_t = (1 - t) * x_0 + t * noise.
        # We do a single large step with the predicted velocity: x_0^hat = z_t - t * v
        return videos_noised - sigma * net_output # [b, t, c, h, w]

#----------------------------------------------------------------------------

class AutoEncoderLoss(EDMLoss):
    def __init__(self, cfg: EasyDict):
        super().__init__(cfg)
        self._init_lpips_models()
        self.teacher = None if all(w == 0 for w in self.cfg.teacher.weights.values()) else load_snapshot(self.cfg.teacher.ckpt, verbose=False)[0].train().requires_grad_(False)

    def compute_ae_loss(self, videos_pred, videos_gt, noise_unscaled, sigma, ctx, cur_step, frames_mask) -> tuple[EasyDict, torch.Tensor, torch.Tensor]:
        targets = self.compute_targets(videos_gt, noise_unscaled, sigma) # [b, t, c, h, w]
        loss_weight = self.compute_loss_weight(sigma) # [b, 1, 1, 1, 1]
        losses_rec_all = self.compute_rec_losses(videos_pred, targets, frames_mask=frames_mask)
        loss_total = loss_weight.reshape(losses_rec_all.rec.shape[0]) * losses_rec_all.rec # [b]
        loss_total, reg_losses_dict = self.maybe_apply_reg(loss_total, ctx, cur_step, loss_weight=loss_weight) # [1], <str, [b]>

        return misc.filter_nones(EasyDict(total=loss_total, **reg_losses_dict, **losses_rec_all))

    def maybe_compute_lipschitz_reg(self, losses: EasyDict, latents: torch.Tensor, videos_pred: torch.Tensor, cur_step: int) -> EasyDict[str, torch.Tensor]:
        _ = cur_step # Unused.
        if self.cfg.dec_lipschitz_reg.r1_weight == 0:
            return losses
        dec_r1_reg = compute_r1_reg(latents, videos_pred, reduce_op='mean') # [b]
        dec_r1_reg_weighted = self.cfg.dec_lipschitz_reg.r1_weight * self.cfg.dec_lipschitz_reg.r1_freq * dec_r1_reg # [b]
        loss_total_new = losses.total + dec_r1_reg_weighted # [b]
        return EasyDict.init_recursively({**losses, 'total': loss_total_new, 'decoder_r1_reg': dec_r1_reg, 'dec_r1_reg_weighted': dec_r1_reg_weighted})


    def maybe_compute_scale_equiv_reg(self, net, losses: EasyDict, latents: torch.Tensor, videos_gt: torch.Tensor, cond, augment_labels) -> EasyDict:
        if self.cfg.scale_equiv_reg.weight == 0:
            return losses

        assert len(self.cfg.scale_equiv_reg.scale_factors) > 0 or self.cfg.scale_equiv_reg.cut_dct2d_high_freqs.max_cut_ratio > 0, "Can only use one of the downsampling methods, but got both."
        assert len(self.cfg.scale_equiv_reg.scale_factors) == 0 or self.cfg.scale_equiv_reg.cut_dct2d_high_freqs.max_cut_ratio == 0, "Expected only one of the downsampling methods to be used, but got both."
        if len(self.cfg.scale_equiv_reg.scale_factors) > 0:
            scale_factor: float = np.random.choice(self.cfg.scale_equiv_reg.scale_factors) # Randomly choose the scale factor.
            if self.cfg.scale_equiv_reg.resample_strategy == 'mean':
                inv_scale_factor = 1.0 / scale_factor
                assert inv_scale_factor.is_integer(), f"Expected inv_scale_factor to be an integer, but got {inv_scale_factor} instead."
                interp_kwargs = dict(pattern='b t c (h fh) (w fw) -> b t c h w', reduction='mean', fh=int(inv_scale_factor), fw=int(inv_scale_factor))
                videos_gt_down, latents_down = [einops.reduce(x, **interp_kwargs) for x in (videos_gt, latents)] # [b, t, c, h/sh, w/sw], [b, lt, c, lh/sh, lw/sw]
            elif self.cfg.scale_equiv_reg.resample_strategy == 'bil':
                interp_kwargs = dict(scale_factor=scale_factor, mode='bilinear', align_corners=True)
                videos_gt_down, latents_down = [F.interpolate(x.flatten(0, 1), **interp_kwargs).unflatten(dim=0, sizes=(len(x), -1)) for x in (videos_gt, latents)] # [b, t, c, h, w]
            else:
                raise NotImplementedError(f"Unknown resampling strategy: {self.cfg.scale_equiv_reg.resample_strategy}")

            # Handling temporal compression separately.
            temporal_scale_factor: float = float(1 / np.random.choice(self.cfg.scale_equiv_reg.temporal_scale_factors)) if len(self.cfg.scale_equiv_reg.temporal_scale_factors) > 0 else 1.0
            if temporal_scale_factor > 1:
                # We resize all the frames except the first one to maintain causality.
                interp_kwargs = dict(pattern='b (t ft) c h w -> b (t ft) c h w', reduction='mean', ft=int(temporal_scale_factor))
                videos_gt_down, latents_down = [torch.cat([x[:, :1], einops.reduce(x[:, 1:], **interp_kwargs)], dim=1) for x in (videos_gt_down, latents_down)] # [b, 1 + t / ft, c, h/sh, w/sw], [b, 1 + lt / ft, c, lh/sh, lw/sw]
        else:
            assert len(self.cfg.scale_equiv_reg.temporal_scale_factors) == 0, f"Expected temporal_scale_factors to be empty, but got {self.cfg.scale_equiv_reg.temporal_scale_factors} instead."
            cut_kwargs = EasyDict(
                cut_ratio=np.random.rand() * self.cfg.scale_equiv_reg.cut_dct2d_high_freqs.max_cut_ratio,
                block_size=self.cfg.scale_equiv_reg.cut_dct2d_high_freqs.block_size,
                zigzag=self.cfg.scale_equiv_reg.cut_dct2d_high_freqs.zigzag,
            )
            # Cut the high frequencies for GT videos and return them together with the cut_kwargs.
            latents_down = cut_dct2d_high_freqs(latents, block_size=cut_kwargs.block_size, zigzag=cut_kwargs.zigzag, cut_ratio=cut_kwargs.cut_ratio) # [b, t, c, h, w]
            rgb_block_size = cut_kwargs.block_size * misc.unwrap_module(net).compression_rate[-1] # [1]
            videos_gt_down = cut_dct2d_high_freqs(videos_gt, block_size=rgb_block_size, zigzag=cut_kwargs.zigzag, cut_ratio=cut_kwargs.cut_ratio) # [b, t, c, h, w]

        videos_pred_down = net(None, None, cond, augment_labels=augment_labels, encode=False, decode=True, latents=latents_down) # [b, t, c, h / sh, w / sw]
        loss_weights_overrides = dict(lpips_pyr=0.0, lpips_vgg=0.0, lpips_alex=0.0, lpips_squeeze=0.0) if self.cfg.scale_equiv_reg.ignore_lpips else None
        ae_down_losses = self.compute_rec_losses(videos_pred_down, videos_gt_down, loss_weights_overrides=loss_weights_overrides) # <str, [b]>
        losses.total = losses.total + self.cfg.scale_equiv_reg.weight * ae_down_losses.rec # [b]
        ae_down_losses = EasyDict({f'latdown_{k}': l for k, l in ae_down_losses.items()}) # <str, [b]>
        return EasyDict.init_recursively({**losses, **ae_down_losses}) # <str, [b]>

    def maybe_compute_highfreq_reg(self, losses: EasyDict, latents: torch.Tensor) -> EasyDict:
        if self.cfg.high_freq_reg.weight == 0:
            return losses

        high_freq_reg = reg_dc_dct2d_high_freqs(latents, self.cfg.high_freq_reg.block_size, self.cfg.high_freq_reg.power) # [batch_size]
        losses.total = losses.total + self.cfg.high_freq_reg.weight * high_freq_reg # [b]
        return EasyDict(**losses, high_freq_reg=high_freq_reg) # <str, [b]>

    def compute_video_pred(self, net, videos_gt, cond, augment_labels, phase) -> tuple[torch.Tensor, torch.Tensor | None, EasyDict]:
        sigma = self.sample_sigma(videos_gt.shape, videos_gt.device) # [b, ...]
        noise_unscaled = torch.randn_like(videos_gt) # [b, t, c, h, w]
        noise_scaled = noise_unscaled * sigma # [b, t, c, h, w]
        videos_aug_noised = self.apply_noise(videos_gt, noise_scaled, sigma, net=net) # [b, t, c, h, w]
        enable_latents_grad = (phase == LossPhase.GenAll and self.cfg.dec_lipschitz_reg.r1_weight > 0) or phase == LossPhase.GenLipReg
        videos_pred, ctx = net(videos_aug_noised, sigma, cond, augment_labels=augment_labels, return_extra_output=True, encode=True, decode=True, enable_latents_grad=enable_latents_grad) # [b, t, c, h, w]
        latents = ctx['latents'] if 'latents' in ctx else None

        return videos_pred, videos_gt, latents, ctx, noise_unscaled, sigma

    def forward(self, net, x: torch.Tensor, cond: TensorGroup, augment_pipe=None,  phase=LossPhase.GenAll, cur_step: int=None, force_sigma_val: Optional[float]=None) -> EasyDict[str, torch.Tensor]:
        assert x.ndim == 5, f"Expected x to be videos and have 5 dimensions: [b, t, c, h, w], got {x.shape} instead."
        _ = force_sigma_val # Unused.
        x, augment_labels = maybe_augment_videos(x, augment_pipe) # [b, t, c, h, w], [b, augment_dim] or None

        if self.cfg.model.is_masked_ae:
            cond.frames_mask = sample_frames_masks(x, self.cfg.model.mask_sampling) # [b, t]
            cond.x_cond = x # [b, t, c, h, w]

        with torch.set_grad_enabled(False if phase == LossPhase.Discr else (self.cfg.dec_lipschitz_reg.r1_weight > 0 or torch.is_grad_enabled())):
            x_pred, x, latents, ctx, noise_unscaled, sigma = self.compute_video_pred(net, x, cond, augment_labels, phase)
            if phase != LossPhase.GenLipReg:
                losses = self.compute_ae_loss(x_pred, x, noise_unscaled, sigma, ctx, cur_step, frames_mask=cond.get('frames_mask')) # [b]
            else:
                losses = EasyDict(total=torch.zeros(len(x), device=x.device)) # <str, [b]>
            losses = self.maybe_compute_scale_equiv_reg(net, losses, latents, x, cond, augment_labels) # <str, [b]>
            losses = self.maybe_compute_highfreq_reg(losses, latents) # <str, [b]>
            if phase in (LossPhase.GenAll, LossPhase.GenLipReg):
                losses = self.maybe_compute_lipschitz_reg(losses, latents, x_pred, cur_step) # <str, [b]>

        return losses

#----------------------------------------------------------------------------

class AlphaFlowLoss(RecFlowLoss):
    @beartype
    def sample_timestep(self, sampling_cfg: EasyDict, cur_step: int, batch_size: int, device: torch.device, upper_truncated: Optional[float] = None) -> torch.Tensor:
        if sampling_cfg.timestep_distrib_type == "logit_norm":
            return sample_logit_norm_time(sampling_cfg, batch_size, device=device)
        elif sampling_cfg.timestep_distrib_type == "truncated_logit_norm":
            assert upper_truncated is not None
            return sample_truncated_logit_norm_time(sampling_cfg, batch_size, device=device, upper_truncated = upper_truncated)
        elif sampling_cfg.timestep_distrib_type == "adaptive_beta":
            # Linearly interpolate alpha and beta from initial to end values between init_steps and end_steps
            if cur_step < sampling_cfg.init_steps:
                alpha, beta = sampling_cfg.initial_alpha, sampling_cfg.initial_beta
            elif cur_step > sampling_cfg.end_steps:
                alpha, beta = sampling_cfg.end_alpha, sampling_cfg.end_beta
            else:
                progress = (cur_step - sampling_cfg.init_steps) / max(1, (sampling_cfg.end_steps - sampling_cfg.init_steps))
                alpha = sampling_cfg.initial_alpha + (sampling_cfg.end_alpha - sampling_cfg.initial_alpha) * progress
                beta = sampling_cfg.initial_beta + (sampling_cfg.end_beta - sampling_cfg.initial_beta) * progress
            beta_distrib = torch.distributions.Beta(torch.tensor([alpha]), torch.tensor([beta]))
            return beta_distrib.sample((batch_size,), device=device)
        elif sampling_cfg.timestep_distrib_type == "uniform":
            return torch.rand((batch_size,), device=device) * (sampling_cfg.max - sampling_cfg.min) + sampling_cfg.min
        elif sampling_cfg.timestep_distrib_type == "constant":
            return torch.ones((batch_size,), device=device) * sampling_cfg.scale
        elif sampling_cfg.timestep_distrib_type == "arctan":
            sigma = torch.exp(torch.randn((batch_size,), device=device) * sampling_cfg.scale + sampling_cfg.location)
            return 2 / math.pi * torch.atan(sigma)

    @beartype
    def get_ratio(self, cfg: EasyDict, cur_step: int) -> float:
        if cfg.scheduler == "constant":
            current_ratio = cfg.initial_value
        elif cfg.scheduler == "step":
            assert cfg.change_init_steps == cfg.change_end_steps, "For step scheduler, change_init_steps and change_end_steps must be equal."
            current_ratio = cfg.initial_value if cur_step < cfg.change_init_steps else cfg.end_value
        elif cfg.scheduler in ["linear", "exponential", "log", "sigmoid"]:
            if cur_step < cfg.change_init_steps:
                current_ratio = cfg.initial_value
            elif cur_step > cfg.change_end_steps:
                current_ratio = cfg.end_value
            else:
                if cfg.scheduler in ["linear", "exponential", "log"]:
                    progress = (cur_step - cfg.change_init_steps) / (cfg.change_end_steps - cfg.change_init_steps)
                elif cfg.scheduler == "sigmoid":
                    middle_step = cfg.change_init_steps + (cfg.change_end_steps - cfg.change_init_steps) / 2
                    progress = (cur_step - middle_step) / (cfg.change_end_steps - cfg.change_init_steps)

                if cfg.scheduler == "linear":
                    current_ratio = cfg.initial_value + (cfg.end_value - cfg.initial_value) * progress
                elif cfg.scheduler == "exponential":
                    progress = progress ** cfg.gamma
                    current_ratio = cfg.initial_value * ((cfg.end_value / cfg.initial_value) ** progress)
                elif cfg.scheduler == "log":
                    log_progress = math.log(1 + progress * 9) / math.log(10)
                    current_ratio = cfg.initial_value + (cfg.end_value - cfg.initial_value) * log_progress
                elif cfg.scheduler == "sigmoid":
                    current_ratio = cfg.initial_value + (cfg.end_value - cfg.initial_value) * (1 / (1 + math.exp(-progress * cfg.gamma)))
        else:
            raise NotImplementedError(f"Unknown scheduler type: {cfg.scheduler}")

        if current_ratio < cfg.clamp_value:
            current_ratio = 0.0
            if "discrete_training" in cfg and cfg.discrete_training:
                current_ratio = cfg.clamp_value
        elif current_ratio > 1 - cfg.clamp_value or (cfg.up_clamp_value is not None and current_ratio > cfg.up_clamp_value):
            current_ratio = 1.0
        return current_ratio

    def sample_timesteps_mf(self, cfg, cur_step, batch_size, device):
        if cfg.type == "truncated":
            t = self.sample_timestep(cfg.time_sampling_mf_t, cur_step, batch_size, device=device)
            t_next = self.sample_timestep(cfg.time_sampling_mf_t_next, cur_step, batch_size, device=device, upper_truncated = t)
        elif cfg.type in ["minmax", "min", "r_in_t_range"]:
            t_1 = self.sample_timestep(cfg.time_sampling_mf_t, cur_step, batch_size, device=device)
            t_2 = self.sample_timestep(cfg.time_sampling_mf_t_next, cur_step, batch_size, device=device)
            if cfg.type == "minmax":
                t = torch.maximum(t_1, t_2)
                t_next = torch.minimum(t_1, t_2)
            elif cfg.type == "min":
                t = t_1
                t_next = torch.minimum(t_1, t_2)
            elif cfg.type == "r_in_t_range":
                t = t_1
                t_next = t_2 * t_1
        else:
            raise NotImplementedError(f"Unknown meanflow distribution type: {cfg.type}")
        return t, t_next

    def sample_traj_params(self, batch_size, cur_step, device):
        ratio_fm = self.get_ratio(self.cfg.ratio_fm, cur_step)
        alpha = self.get_ratio(self.cfg.alpha, cur_step)
        batch_size_fm = int(batch_size * ratio_fm)
        batch_size_mf = batch_size - batch_size_fm

        t_fm = t_next_fm = self.sample_timestep(self.cfg.time_sampling_fm, cur_step, batch_size_fm, device=device)
        dt_fm = torch.zeros_like(t_next_fm)
        t_mf, t_next_mf = self.sample_timesteps_mf(self.cfg.distrib_t_t_next_mf, cur_step, batch_size_mf, device)
        dt_mf = alpha * (t_mf - t_next_mf)

        t = torch.cat([t_fm, t_mf], dim=0)
        t_next = torch.cat([t_next_fm, t_next_mf], dim=0)
        dt = torch.cat([dt_fm, dt_mf], dim=0)

        return t.view(batch_size, 1, 1, 1, 1), t_next.view(batch_size, 1, 1, 1, 1), dt.view(batch_size, 1, 1, 1, 1), alpha

    @torch.no_grad()
    def _compute_velocity_cfg(self, velocity, x_t, t, cond, augment_labels, net, batch_size):
        # Create classifier free guidance mask for t: True where self.cfg.cfg_params.t_min < t < self.cfg.cfg_params.t_max
        t_flat = t.view(batch_size, -1)
        omega = self.cfg.cfg_params.omega
        kappa = self.cfg.cfg_params.kappa
        mask = (t_flat > self.cfg.cfg_params.t_min) & (t_flat < self.cfg.cfg_params.t_max)
        cfg_mask_idx = mask.view(batch_size).bool()  # [b]
        # Only apply classifier free guidance t is in the interval and when scale != 1.0
        velocity_cfg = velocity.clone() # [b, t, c, h, w]

        # Drop labels, with probability self.cfg.label_dropout, set cond.label[...] to zero vector
        if self.cfg.model.label_dropout > 0.0:
            label_drop_mask_idx = (torch.rand(cond.label.shape[0], device=cond.label.device) < self.cfg.model.label_dropout)
            drop_mask = cfg_mask_idx & label_drop_mask_idx
            cond.label[drop_mask] = torch.zeros_like(cond.label[drop_mask])

        if 1 - omega - kappa != 0.0:
            videos_u_t_t_uncond = net(
                x_t, sigma_next=t, sigma=t,
                cond=None, augment_labels=augment_labels, return_extra_output=False
            ) # [b, t, c, h, w]
        else:
            videos_u_t_t_uncond = torch.zeros_like(x_t) # [b, t, c, h, w]

        if kappa != 0:
            videos_u_t_t_cond = net(
                x_t, sigma_next=t, sigma=t,
                cond=cond, augment_labels=augment_labels, return_extra_output=False
            ) # [b, t, c, h, w]
        else:
            videos_u_t_t_cond = torch.zeros_like(x_t) # [b, t, c, h, w]

        guided = omega * velocity + kappa * videos_u_t_t_cond + (1 - omega - kappa) * videos_u_t_t_uncond # [b, t, c, h, w]
        velocity_cfg[cfg_mask_idx] = guided[cfg_mask_idx] # [b, t, c, h, w]
        return velocity_cfg

    @torch.no_grad()
    def _compute_mean_velocity_c(self, x_t, t_next, t, velocity_cfg, cond, augment_labels, net):
        if x_t.shape[0] == 0:
            return torch.empty((0, *velocity_cfg.shape[1:]), device=velocity_cfg.device)
        t = t.flatten() # [b]
        t_next = t_next.flatten() # [b]

        mask_mf = ~torch.isclose(t_next, t) # [b]
        batch_size_mf = mask_mf.sum().item()
        mean_velocity = velocity_cfg.clone() # [b, t, c, h, w]

        def wrap_net(x_t, t_next, t):
            return net(x_t, sigma_next=t_next, sigma=t, cond=cond[mask_mf], augment_labels=augment_labels, return_extra_output=False)
        x_t_mf, t_mf, t_next_mf, velocity_cfg_mf = x_t[mask_mf], t[mask_mf], t_next[mask_mf], velocity_cfg[mask_mf]

        if batch_size_mf == 0:
            return mean_velocity

        _, videos_dudt_mf = jvp(
            wrap_net,
            (x_t_mf, t_next_mf, t_mf),
            (velocity_cfg_mf, torch.zeros_like(t_next_mf), torch.ones_like(t_mf)),
        )
        mean_velocity_mf = (velocity_cfg_mf -  (t_mf - t_next_mf).view(batch_size_mf, 1, 1, 1, 1) * videos_dudt_mf) # [b_mf, t, c, h, w]
        mean_velocity[mask_mf] = mean_velocity_mf
        return mean_velocity # [b, t, c, h, w]

    @torch.no_grad()
    def _compute_mean_velocity_d(self, x_t, velocity_cfg, t_next, t, dt, cond, augment_labels, net):
        if x_t.shape[0] == 0:
            return torch.empty((0, *velocity_cfg.shape[1:]), device=velocity_cfg.device)
        x_t_minus_dt = x_t - dt * velocity_cfg # [b, t, c, h, w]

        if torch.isclose(1 - dt/(t - t_next), torch.zeros_like(t)).all():
            mean_velocity_next = torch.zeros_like(x_t_minus_dt) # [b, t, c, h, w]
        else:
            mean_velocity_next, _ = net(x_t_minus_dt, sigma_next = t_next, sigma = t - dt, cond=cond, augment_labels=augment_labels, return_extra_output=True) # [b, t, c, h, w]

        mean_velocity = (dt * velocity_cfg + (t - dt - t_next) * mean_velocity_next) / (t - t_next) # [b, t, c, h, w]
        mean_velocity = torch.clip(mean_velocity, min=-self.cfg.clamp_utgt, max=self.cfg.clamp_utgt) # [b, t, c, h, w]
        return mean_velocity # [b, t, c, h, w]


    def forward(self, net, x, cond, augment_pipe=None, phase=LossPhase.Gen, cur_step=None, force_t_val: Optional[float]=None, force_t_r_dt_val=None, align_ctx=None, compute_decomposed_loss=False) -> EasyDict[str, torch.Tensor]:
        _ = force_t_val ## Unused variable
        assert x.ndim == 5, f"Expected x to have 5 dimensions: [b, t, c, h, w], got {x.shape} instead."
        batch_size = len(x) # [1]
        x_aug, augment_labels = maybe_augment_videos(x, augment_pipe) # [b, t, c, h, w], [b, augment_dim] or None
        assert augment_labels is None, "Augment labels are not supported for joint training"
        x_aug = self._maybe_encode(net, x_aug, cond) # [b, t, c, h, w]

        # Sample t, t_next and alpha, dt = (t - t_next) * alpha
        t, t_next, dt, alpha = self.sample_traj_params(x_aug.shape[0], cur_step, x_aug.device) # [b, 1, 1, 1, 1], [b, 1, 1, 1, 1]
        noise_unscaled = torch.randn_like(x_aug) # [b, t, c, h, w]
        noise_scaled = noise_unscaled * t # [b, t, c, h, w]
        x_t = self.apply_noise(x_aug, noise_scaled, t, net) # [b, t, c, h, w]
        velocity = noise_unscaled - x_aug # [b, t, c, h, w]

        # Apply cfg training
        velocity_cfg = self._compute_velocity_cfg(velocity=velocity, x_t=x_t, t=t, cond=cond, augment_labels=augment_labels, net=net, batch_size=batch_size) # [b, t, c, h, w]

        # Split batch for continuous (alpha == 1 or r == t) and discrete training (0 < alpha <= 1)
        mask_c = (dt == 0).flatten() # [b]
        mask_d = ~mask_c # [b]
        batch_size_c, batch_size_d = mask_c.sum().item(), mask_d.sum().item() # [1], [1]
        velocity_cfg_c, x_t_c, t_c, t_next_c, cond_c, = velocity_cfg[mask_c], x_t[mask_c], t[mask_c], t_next[mask_c], cond[mask_c]
        x_t_d, velocity_cfg_d, t_d, t_next_d, dt_d, cond_d, = x_t[mask_d], velocity_cfg[mask_d], t[mask_d], t_next[mask_d], dt[mask_d], cond[mask_d]

        # Calculate u_tgt when alpha == 1 or r == t
        mean_velocity_c = self._compute_mean_velocity_c(x_t_c, t_next_c, t_c, velocity_cfg_c, cond_c, augment_labels, net) # [b_c, t, c, h, w]

        # Calculate u_tgt when 0 < alpha <= 1
        mean_velocity_d = self._compute_mean_velocity_d(x_t_d, velocity_cfg_d, t_next_d, t_d, dt_d, cond_d, augment_labels, net) # [b_d, t, c, h, w]

        mean_velocity = torch.cat([mean_velocity_c, mean_velocity_d], dim=0) # [b, t, c, h, w]

        pred_mean_velocity, ctx = net(
            x_t,
            sigma_next=t_next,
            sigma=t,
            cond=cond,
            augment_labels=augment_labels,
            return_extra_output=True
        ) # [b, t, c, h, w]

        ## Adaptive loss
        loss_unscaled = ((pred_mean_velocity - mean_velocity) ** 2).flatten(1).mean(1) # [b]
        weight_c = torch.ones(batch_size_c, device=velocity.device) # [b_c]
        weight_d = torch.ones(batch_size_d, device=velocity.device) * alpha # [b_d]
        weight = torch.cat([weight_c, weight_d], dim=0) / (loss_unscaled.detach() + self.cfg.adaptive_loss_weight_eps) # [b]
        loss = weight * loss_unscaled # [b]

        ## Compute trajectory flow matching loss
        loss_tfm = ((pred_mean_velocity - velocity_cfg) ** 2).flatten(1).mean(1) # [b]

        ## Compute consistency flow matching loss
        loss_tcc = (2 * (velocity_cfg - mean_velocity) * pred_mean_velocity).flatten(1).mean(1) # [b]
        loss_tfm_plus_tcc = loss_tfm + loss_tcc # [b]

        loss_dict = misc.filter_nones(EasyDict(
            total                      = loss, # [b]
            trajectory_FM              = loss_tfm, # [b]
            trajectory_consistency     = loss_tcc, # [b]
            trajectory_sum             = loss_tfm_plus_tcc, # [b]
            ))
        return loss_dict

#----------------------------------------------------------------------------
# Various reconstruction loss functions.

def reweigh_supp_loss(loss_main: torch.Tensor, loss_supp: torch.Tensor, relative_weight: float, eps: float=1e-6) -> torch.Tensor:
    # Re-adjusts the loss_supp to have the magnitude of relative_weight * loss_main.
    assert relative_weight > 0, f"Expected relative_weight to be positive, but got {relative_weight} instead."
    return loss_supp * relative_weight * (loss_main.abs().mean().item() / (loss_supp.abs().mean().item() + eps)) # [<loss_supp shape>]

def compute_image_loss_per_frame(x_rec: torch.Tensor, x_gt: torch.Tensor, loss_fn: Callable, *args, **kwargs) -> torch.Tensor:
    assert x_rec.ndim == 5, f"Expected x to have 5 dimensions: [b, t, c, h, w], got {x_rec.shape} instead."
    assert x_rec.shape == x_gt.shape, f"Expected x_rec and x_gt to have the same shape, but got {x_rec.shape} and {x_gt.shape} instead."
    b, t = x_rec.shape[:2] # (1, 1)
    x_rec = einops.rearrange(x_rec, 'b t c h w -> (b t) c h w') # [b * t, c, h, w]
    x_gt = einops.rearrange(x_gt, 'b t c h w -> (b t) c h w') # [b * t, c, h, w]
    loss = loss_fn(x_rec, x_gt, *args, **kwargs) # [b * t]
    loss = loss.view(-1, 1, 1, 1) if loss.ndim == 1 else loss # [b * t, 1, 1, 1]
    loss = einops.rearrange(loss, '(b t) 1 1 1 -> b t 1 1 1', b=b, t=t) # [b, t, 1, 1, 1]
    return loss # [b, t, 1, 1, 1]

def compute_image_lpips(x_rec: torch.Tensor, x_gt: torch.Tensor, lpips_fn: Callable, loss_cfg: DictConfig) -> torch.Tensor:
    num_tiles = 1
    if loss_cfg.downsample_to_native:
        if loss_cfg.downsample_tiled:
            # Split the image into tiles, compute LPIPS for each tile, and average the results.
            x_rec, x_gt = [pad_to_divisible(x, 224) for x in (x_rec, x_gt)] # 2 x [b, c, h' * 224, w' * 224]
            x_rec, x_gt = [F.unfold(x, kernel_size=224, stride=224) for x in (x_rec, x_gt)] # 2 x [b, c * 224 * 224, num_tiles]
            num_tiles = x_rec.shape[-1] # (1)
            x_rec, x_gt = [einops.rearrange(x, 'b (c h w) n -> (b n) c h w', c=3, h=224, w=224) for x in (x_rec, x_gt)]
        else:
            x_rec = F.interpolate(x_rec, size=(224, 224), mode='area') # [b, c, 224, 224]
            x_gt = F.interpolate(x_gt, size=(224, 224), mode='area') # [b, c, 224, 224]
    loss_total, losses_per_layer = lpips_fn(x_rec, x_gt, retPerLayer=True) # [b, 1, 1, 1]
    loss = loss_total if loss_cfg.num_first_layers is None else torch.stack(losses_per_layer[:loss_cfg.num_first_layers]).sum(dim=0) # [b, 1, 1, 1]

    if loss_cfg.downsample_to_native and loss_cfg.downsample_tiled:
        loss = einops.rearrange(loss, '(b n) 1 1 1 -> b n 1 1 1', n=num_tiles).mean(dim=1) # [b, 1, 1, 1]
    return loss

def compute_video_lpips(x_rec: torch.Tensor, x_gt: torch.Tensor, lpips_fn: Callable, *args, **kwargs) -> torch.Tensor:
    return compute_image_loss_per_frame(x_rec, x_gt, compute_image_lpips, lpips_fn, *args, **kwargs) # [b, t, 1, 1, 1]

def compute_img_grad_loss_per_frame(x_rec: torch.Tensor, x_gt: torch.Tensor) -> torch.Tensor:
    return compute_image_loss_per_frame(x_rec, x_gt, compute_img_grad_loss) # [b, t, 1, 1, 1]

def compute_img_grad_loss(x_rec: torch.Tensor, x_gt: torch.Tensor) -> torch.Tensor:
    x_rec_grad = compute_img_grad_magnitude(x_rec) # [b, h, w]
    x_gt_grad = compute_img_grad_magnitude(x_gt) # [b, h, w]
    loss = (x_rec_grad - x_gt_grad).pow(2).mean(dim=[1, 2]).sqrt() # [b]
    return loss

def compute_img_grad_magnitude(x: torch.Tensor, filter_pt: torch.Tensor=None) -> torch.Tensor:
    assert x.ndim == 4, f"Expected x to have 4 dimensions: [b, c, h, w], got {x.shape} instead."
    filter_pt = torch.tensor(SOBEL_FILTER, dtype=torch.float32, device=x.device).reshape(2, 1, 3, 3) if filter_pt is None else filter_pt # [2, 1, 3, 3]
    batch_size, num_channels = x.shape[0], x.shape[1] # (1)
    x = einops.rearrange(x, 'b c h w -> (b c) 1 h w') # [b * c, 1, h, w]
    grad = F.conv2d(x, filter_pt, padding=1) # [b * c, 2, h, w], pylint: disable=not-callable
    grad_magnitude = grad.pow(2).sum(dim=1).sqrt() # [b * c, h, w]
    grad_magnitude = einops.rearrange(grad_magnitude, '(b c) h w -> b c h w', b=batch_size, c=num_channels) # [b, c, h, w]
    grad_magnitude = grad_magnitude.mean(dim=1) # [b, h, w]

    return grad_magnitude

def compute_video_freq3d_loss(x_rec: torch.Tensor, x_gt: torch.Tensor) -> torch.Tensor:
    assert x_rec.ndim == 5, f"Expected x to have 5 dimensions: [b, t, c, h, w], got {x_rec.shape} instead."
    assert x_rec.shape == x_gt.shape, f"Expected x_rec and x_gt to have the same shape, but got {x_rec.shape} and {x_gt.shape} instead."
    return compute_freq_loss(x_rec, x_gt, dim=(1, 3, 4)).view(-1, 1, 1, 1, 1) # [b, 1, 1, 1, 1]

def compute_video_freq2d_loss(x_rec: torch.Tensor, x_gt: torch.Tensor) -> torch.Tensor:
    return compute_image_loss_per_frame(x_rec, x_gt, compute_img_freq_loss) # [b, t, 1, 1, 1]

def compute_img_freq_loss(x_rec: torch.Tensor, x_gt: torch.Tensor) -> torch.Tensor:
    return compute_freq_loss(x_rec, x_gt, dim=(2, 3)).view(-1, 1, 1, 1) # [b, 1, 1, 1]

def compute_freq_loss(x_rec, x_gt, dim: tuple[int]):
    # Transform both inputs to the frequency domain using FFT
    fft_rec = torch.fft.fftn(x_rec, dim=dim) # pylint: disable=not-callable
    fft_gt = torch.fft.fftn(x_gt, dim=dim) # pylint: disable=not-callable

    # Shift the zero frequency component to the center (TODO: we don't really need that?)
    fft_shifted_rec = torch.fft.fftshift(fft_rec, dim=dim) # pylint: disable=not-callable
    fft_shifted_gt = torch.fft.fftshift(fft_gt, dim=dim) # pylint: disable=not-callable

    # Compute magnitude and/or phase
    magnitude_rec = torch.abs(fft_shifted_rec) # [b, c, d1, ..., dn]
    magnitude_gt = torch.abs(fft_shifted_gt) # [b, c, d1, ..., dn]

    # Compute phase
    # phase_rec = torch.angle(gen_fft_shift)
    # phase_gt = torch.angle(gt_fft_shift)

    loss = (magnitude_rec - magnitude_gt).pow(2).view(magnitude_rec.shape[0], -1).mean(dim=1).sqrt() # [b]

    return loss

def compute_image_random_conv_l2(x_rec: torch.Tensor, x_gt: torch.Tensor, loss_cfg: DictConfig) -> torch.Tensor:
    p: int = loss_cfg.patch_size # [1]
    weight = torch.randn(loss_cfg.embed_dim, x_rec.shape[1], p, p, device=x_rec.device, dtype=x_rec.dtype) / (x_rec.shape[1] * p ** 2) # [embed_dim, c, kh, kw]
    x_rec, x_gt = [einops.rearrange(F.conv2d(x, weight, stride=p // 2, padding=0), 'b c h w -> b (h w) c') for x in (x_rec, x_gt)] # pylint: disable=not-callable
    return (x_rec - x_gt).pow(2).flatten(start_dim=1).mean(dim=1).sqrt().view(-1, 1, 1, 1) # [b, 1, 1, 1]

def compute_video_random_conv_l2(x_rec: torch.Tensor, x_gt: torch.Tensor, loss_cfg: DictConfig) -> torch.Tensor:
    assert x_rec.shape == x_gt.shape, f"Expected x_rec and x_gt to have the same shape, but got {x_rec.shape} and {x_gt.shape} instead."
    x_rec, x_gt = [einops.rearrange(x, 'b t c h w -> b c t h w') for x in (x_rec, x_gt)] # 2 x [b, c, t, h, w]
    t, h, w = x_rec.shape[2:] # 3x (1,)
    kernel_size = tuple(min(r, loss_cfg.patch_size) for r in (t, h, w)) # [3]
    stride = tuple(max(1, k // 2) for k in kernel_size) # [3]
    padding = tuple(k // 2 for k in kernel_size) # [3]
    weight = torch.randn(loss_cfg.embed_dim, x_rec.shape[1], *kernel_size, device=x_rec.device, dtype=x_rec.dtype) / (x_rec.shape[1] * np.prod(kernel_size)) # [embed_dim, c, kt, kh, kw]
    x_rec, x_gt = [F.conv3d(x, weight, stride=stride, padding=padding) for x in (x_rec, x_gt)] # 2 x [b, embed_dim, t', h', w']; pylint: disable=not-callable
    loss = (x_rec - x_gt).pow(2).flatten(start_dim=1).mean(dim=1).sqrt().view(-1, 1, 1, 1, 1) # [b, 1, 1, 1, 1]
    return loss

def compute_framewise_video_random_conv_l2(x_rec: torch.Tensor, x_gt: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    return compute_image_loss_per_frame(x_rec, x_gt, compute_image_random_conv_l2, *args, **kwargs) # [b, t, 1, 1, 1]

#----------------------------------------------------------------------------
# Diffusion training utility functions.

def sample_logit_norm_time(time_sampling_cfg: EasyDict, shape: torch.Size | tuple, device: torch.device=None) -> torch.Tensor:
    """
    Time Samples following the Logit Normal distribution of Stable Diffusion 3
    Produces times in [0, 1-eps] following "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow"
    """
    randn = misc.randn(shape, device=device) * time_sampling_cfg.scale + time_sampling_cfg.location # [b, ...]
    logit_normal = randn.sigmoid() # [b, ...]
    logit_normal_rescaled = logit_normal * (1 - time_sampling_cfg.eps) # [b, ...]. Rescales between [0, 1-eps]

    return logit_normal_rescaled

def sample_truncated_logit_norm_time(time_sampling_cfg: EasyDict, batch_size: int, device: torch.device=None, upper_truncated = None) -> torch.Tensor:
    """
    Time Samples following the Logit Normal distribution of Stable Diffusion 3
    Produces times in [0, 1-eps] following "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow"
    """
    from torchrl.modules.distributions import TruncatedNormal
    assert upper_truncated is not None, "upper_truncated must be specified"
    loc = torch.ones(batch_size, device=device) * time_sampling_cfg.location # [b]
    scale = torch.ones(batch_size, device=device) * time_sampling_cfg.scale # [b]
    trunc_randn = TruncatedNormal(loc=loc, scale=scale, low=-float('Inf'), high=torch.logit(upper_truncated)).sample() # [b]
    logit_normal = torch.sigmoid(trunc_randn) # [b]

    return logit_normal

def sample_custom_t_steps(t_steps: list[float], shape: torch.Size | int, device) -> torch.Tensor:
    """Sampling uniformly from a list of t_steps."""
    assert len(t_steps) > 0, f"Expected t_steps to be non-empty, but got {t_steps} instead."
    assert all(0.0 <= t <= 1.0 for t in t_steps), f"Expected all t_steps to be in [0, 1], but got {t_steps} instead."
    t_steps_sampled = np.random.choice(t_steps, size=tuple(shape), replace=True) # [b, ...]
    t_steps_sampled = torch.from_numpy(t_steps_sampled).float().to(device) # [b, ...]

    return t_steps_sampled

#----------------------------------------------------------------------------
# Utility functions.

def _maybe_convert_tensor_group(losses: dict[str, TensorLike], sum_key='total', default_modality='video') -> EasyDict:
    """We can't backprop through TensorGroup losses, so we convert/rename them to tensors here."""
    out = {}
    for k, v in losses.items():
        if isinstance(v, TensorGroup):
            if k == sum_key:
                out[sum_key] = torch.stack(list(v.flatten(1).mean(1).values()), dim=1).sum(dim=1) # [b]
            else:
                for modality in v.keys():
                    if modality == default_modality:
                        out[k] = v.video
                    out[f"{k}_{modality}"] = v[modality]

        else:
            out[k] = v
    return EasyDict(out)

def maybe_filter_loss_by_mask(loss_rec: torch.Tensor, frames_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
    assert loss_rec.ndim == 5, f"Expected loss_rec to have 5 dimensions: [b, t, c, h, w], got {loss_rec.shape} instead."
    assert frames_mask is None or frames_mask.ndim == 2, f"Expected frames_mask to have 2 dimensions: [b, t], got {frames_mask.shape} instead."
    assert frames_mask is None or len(loss_rec) == len(frames_mask), f"Expected loss_rec and frames_mask to have the same batch size, but got {len(loss_rec)} and {len(frames_mask)} instead."
    if frames_mask is not None:
        # Do not compute the loss for MISSING and CONTEXT frames.
        assert frames_mask.shape[:2] == loss_rec.shape[:2], f"Expected frames_mask and loss to have the same shape, but got {frames_mask.shape} and {loss_rec.shape} instead."
        keep_mask = frames_mask == TokenType.QUERY.value # [b, t]
        loss_rec = [l[m] for l, m in zip(loss_rec, keep_mask)] # (batch_size, [<any>, c, h, w])
        assert all(l.shape[0] > 0 for l in loss_rec), f"Expected all losses to have at least one frame, but got {list(map(len, loss_rec))} frames instead."
        loss_rec = torch.stack([l.mean(dim=0) for l in loss_rec]).unsqueeze(1) # [b, 1, c, h, w]
    return loss_rec

def pad_to_divisible(x: torch.Tensor, divisor: int=224) -> torch.Tensor:
    _b, _c, h, w = x.shape
    pad_h_total = (divisor - h % divisor) % divisor
    pad_w_total = (divisor - w % divisor) % divisor
    pad_top, pad_left = pad_h_total // 2, pad_w_total // 2
    pad_bottom, pad_right = pad_h_total - pad_top, pad_w_total - pad_left # Residual padding
    x_padded = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom)) # [b, c, h', w']

    return x_padded

def maybe_augment_videos(videos: torch.Tensor, augment_pipe: Optional[torch.nn.Module]):
    if not augment_pipe is None:
        num_frames = videos.shape[1]
        videos = einops.rearrange(videos, 'b t c h w -> b (t c) h w') # [b, t * c, h, w]
        videos_aug, augment_labels = augment_pipe(videos, num_frames=num_frames) # [b, t * c, h, w], [b, augment_dim]
        videos_aug = einops.rearrange(videos_aug, 'b (t c) h w -> b t c h w', t=num_frames) # [b, t, c, h, w]
    else:
        videos_aug, augment_labels = (videos, None)
    return videos_aug, augment_labels

#----------------------------------------------------------------------------
# Lipschitz regularization utilities.

def compute_r1_reg(x: torch.Tensor, y: torch.Tensor, reduce_op: str='sum') -> torch.Tensor:
    assert x.ndim == 5, f"Expected x to have 5 dimensions: [b, t, c, h, w], got {x.shape} instead."
    assert x.requires_grad, "Expected x to require gradients."
    r1_grads = torch.autograd.grad(outputs=[y.sum()], inputs=[x], create_graph=True, only_inputs=True)[0] # [b * np, t, c, h, w]
    r1_penalty = r1_grads.square() # [b * np, t, c, h, w]
    r1_penalty = einops.reduce(r1_penalty, 'b t c h w -> b', reduce_op) # [b * np]

    return r1_penalty

#----------------------------------------------------------------------------
