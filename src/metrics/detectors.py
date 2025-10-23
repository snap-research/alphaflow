import os
import pickle
from typing import Callable
from urllib.parse import urlparse
from collections import namedtuple
from loguru import logger

import torch
import torchvision.transforms.functional as TVF
import einops
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, LearnedPerceptualImagePatchSimilarity
try:
    import torchaudio
except ImportError:
    torchaudio = None

from src.utils import misc, distributed as dist, os_utils

#----------------------------------------------------------------------------

def get_feature_detector_name(url):
    return os.path.splitext(url.split('/')[-1])[0]

def load_feature_detector(url, device=torch.device('cpu'), verbose=False) -> Callable:
    with dist.leader_first(), os_utils.open_path(url, verbose=(verbose and dist.is_main_process())) as f:
        detector = pickle.load(f) if urlparse(url).path.endswith('.pkl') else torch.jit.load(f).eval()
    return detector.to(device)

#----------------------------------------------------------------------------

# TODO: rename `detector` into a feature extractor (including all the relevant arguments).
Detector = namedtuple('Detector', ['run', 'name'])

def construct_inception_detector(device: torch.device, verbose: bool=False) -> Detector:
    # Direct TorchScript translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt'
    name = get_feature_detector_name(url)
    detector: torch.nn.Module = load_feature_detector(url=url, device=device, verbose=verbose)

    @torch.autocast(device_type="cuda", enabled=False)
    def run_detector(videos: torch.Tensor, resize_frames: tuple[int]=None, text: list[str]=None, videos_gt: torch.Tensor=None, waveform: torch.Tensor=None) -> torch.Tensor: # pylint: disable=unused-argument
        b, t = videos.shape[:2]
        # Pick only a single random frame out of each video.
        videos = videos[torch.arange(b), torch.randint(t, size=(b,))].unsqueeze(1) # [b, 1, c, h, w]
        videos = rearrange_videos(videos, resize=resize_frames, temporal_detector=False) # [b, c, h, w]
        features = detector(videos, return_features=True) # [b, d]
        return features

    return Detector(run_detector, name)

def construct_inception3d_detector(device: torch.device, verbose: bool=False) -> Detector:
    # Reproduced torchscript version of the I3D model, trained on Kinetics-400, used here:
    # https://github.com/google-research/google-research/blob/master/frechet_video_distance/frechet_video_distance.py
    # Note that the weights on tf.hub (used in the script above) differ from the original released weights
    url = 'https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1'
    name = get_feature_detector_name(url)
    detector: torch.nn.Module = load_feature_detector(url=url, device=device, verbose=verbose)

    @torch.autocast(device_type="cuda", enabled=False)
    def run_detector(videos: torch.Tensor, resize_frames: tuple[int]=None, text: list[str]=None, videos_gt: torch.Tensor=None, waveform: torch.Tensor=None) -> torch.Tensor: # pylint: disable=unused-argument
        _b, t, _c, h, w = videos.shape
        t_new, h_new, w_new = (max(t, 16), h, w) if resize_frames is None else (max(t, 16), resize_frames[0], resize_frames[1])
        resize_video = (t_new, h_new, w_new) if (t != t_new or h != h_new or w != w_new) else None
        videos = rearrange_videos(videos, resize=resize_video, temporal_detector=True) # [b, c, t, h, w]
        features = detector(videos, rescale=True, resize=True, return_features=True) # [b, d]
        return features

    return Detector(run_detector, name)

def construct_legacy_inception_detector(device: torch.device, verbose: bool=False) -> Detector:
    # Legacy pkl inception detector for InceptionScore (following the logic from StyleGAN3).
    # TODO: check if the new torchscript version is compatible.
    # Direct TorchScript translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    name = get_feature_detector_name(url)
    detector: torch.nn.Module = load_feature_detector(url=url, device=device, verbose=verbose)

    @torch.autocast(device_type="cuda", enabled=False)
    def run_detector(videos: torch.Tensor, resize_frames: tuple[int]=None, text: list[str]=None, videos_gt: torch.Tensor=None, waveform: torch.Tensor=None) -> torch.Tensor:
        _ = text, videos_gt # Unused.
        b, t = videos.shape[:2]
        # Pick only a single random frame out of each video.
        videos = videos[torch.arange(b), torch.randint(t, size=(b,))].unsqueeze(1) # [b, 1, c, h, w]
        videos = rearrange_videos(videos, resize=resize_frames, temporal_detector=False) # [b, c, h, w]
        features = detector(videos, no_output_bias=True) # [b, c]
        return features

    return Detector(run_detector, name)

def construct_video_inception_detector(device: torch.device, verbose: bool=False) -> Detector:
    # Legacy pkl inception detector for InceptionScore (following the logic from StyleGAN3).
    # TODO: check if the new torchscript version is compatible.
    # Direct TorchScript translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    url = 'https://www.dropbox.com/s/jxpu7avzdc9n97q/c3d_ucf101.pt?dl=1'
    name = get_feature_detector_name(url)
    detector: torch.nn.Module = load_feature_detector(url=url, device=device, verbose=verbose)

    @torch.autocast(device_type="cuda", enabled=False)
    def run_detector(videos: torch.Tensor, resize_frames: tuple[int]=None, text: list[str]=None, videos_gt: torch.Tensor=None, waveform: torch.Tensor=None) -> torch.Tensor:
        _ = text, videos_gt # Unused.
        _b, t, _c, h, w = videos.shape
        t_new, h_new, w_new = (max(t, 16), h, w) if resize_frames is None else (max(t, 16), resize_frames[0], resize_frames[1])
        resize_video = (t_new, h_new, w_new) if (t != t_new or h != h_new or w != w_new) else None
        videos = rearrange_videos(videos, resize=resize_video, temporal_detector=True) # [b, c, t, h, w]
        features = detector(videos) # [b, c]
        return features

    return Detector(run_detector, name)

def construct_dinov2_detector(device: torch.device, verbose: bool=False) -> Detector:
    name = 'dinov2_vitl14'
    detector: torch.nn.Module = torch.hub.load('facebookresearch/dinov2:main', name, trust_repo=True, verbose=verbose, skip_validation=True)
    detector.eval().requires_grad_(False).to(device)

    @torch.autocast(device_type="cuda", enabled=False)
    def run_detector(videos: torch.Tensor, resize_frames: tuple[int]=None, text: list[str]=None, videos_gt: torch.Tensor=None, waveform: torch.Tensor=None) -> torch.Tensor:
        _ = text, videos_gt # Unused.
        b, t = videos.shape[:2]
        # Pick only a single random frame out of each video.
        random_frames = videos[torch.arange(b), torch.randint(t, size=(b,))].unsqueeze(1) # [b, 1, c, h, w]
        random_frames = rearrange_videos(random_frames, resize=resize_frames, temporal_detector=False) # [b, c, h, w]
        random_frames = torch.nn.functional.interpolate(random_frames.to(torch.float32), size=(224, 224), mode='bicubic', antialias=True) # [b, c, 224, 224]

        random_frames = random_frames.to(torch.float32) / 255 # [b, c, 224, 224]
        random_frames = random_frames - misc.const_like(random_frames, [0.485, 0.456, 0.406]).reshape(1, -1, 1, 1) # [b, c, 224, 224]
        random_frames = random_frames / misc.const_like(random_frames, [0.229, 0.224, 0.225]).reshape(1, -1, 1, 1) # [b, c, 224, 224]

        features = detector(random_frames)
        assert features.ndim == 2, f"Wrong output features shape: {features.shape}"

        return features

    return Detector(run_detector, name)

#----------------------------------------------------------------------------

def construct_clip_detector(device: torch.device, verbose: bool=False, return_embeddings: str | None=None) -> Detector:
    assert return_embeddings in (None, 'image', 'text'), f'Invalid return_embeddings: {return_embeddings}'
    _ = verbose # Unused
    import clip # pylint: disable=import-outside-toplevel

    vis_encoder_name = 'ViT-B/32'
    with dist.leader_first():
        clip_model, preprocess = clip.load(vis_encoder_name, download_root=os.environ.get('TORCH_HOME', None))
    clip_model.to(device)
    name = f'clip-{vis_encoder_name}'.replace("/", "-")

    @torch.autocast(device_type="cuda", enabled=False)
    def run_detector(videos: torch.Tensor, resize_frames: tuple[int]=None, text: list[str]=None, videos_gt: torch.Tensor=None, waveform: torch.Tensor=None) -> torch.Tensor:
        assert text is not None or return_embeddings == 'image', 'text must be provided for CLIP score.'
        assert return_embeddings == 'image' or len(videos) == len(text), f'Wrong number of videos: {len(videos)} vs {len(text)}'
        _ = videos_gt # Unused.
        _b, t, _c, h, w = videos.shape
        _t_new, h_new, w_new = (max(t, 16), h, w) if resize_frames is None else (max(t, 16), resize_frames[0], resize_frames[1])
        resize_frames = (h_new, w_new) if (h != h_new or w != w_new) else None # Not using t_new since we are extracting a single frame anyways.
        videos = videos[:, [videos.shape[1] // 2]] # Selecting the middle frame.
        videos = rearrange_videos(videos, resize=resize_frames, temporal_detector=False) # [b, c, h, w]

        # Resizing the frames separately.
        images = torch.stack([preprocess(TVF.to_pil_image(x)) for x in videos]).to(device) # [b, c, h, w]
        image_features = clip_model.encode_image(images) # [b, d]
        if return_embeddings == 'image':
            return image_features
        text_tokenized = clip.tokenize(text, truncate=True).to(device) # [b, 77]
        text_features = clip_model.encode_text(text_tokenized).to(image_features.dtype) # [b, d]
        if return_embeddings == 'text':
            return text_features

        assert image_features.ndim == 2, f"Wrong shape: {image_features.shape}"
        assert text_features.ndim == 2, f"Wrong shape: {text_features.shape}"
        assert image_features.shape == text_features.shape, f"Wrong shape: {image_features.shape} vs {text_features.shape}"

        image_features = image_features / image_features.norm(dim=1, keepdim=True) # [b, d]
        text_features = text_features / text_features.norm(dim=1, keepdim=True) # [b, d]
        similarities = (image_features * text_features).sum(dim=1, keepdim=True) # [b, 1]

        return similarities

    return Detector(run_detector, name)

#----------------------------------------------------------------------------
# A reconstruction detector, which computes L1, L2, LPIPS, SSIM and PSNR scores.
# All the metrics are image-based, unfortunately...

def construct_reconstruction_detector(device: torch.device) -> Detector:
    with dist.leader_first():
        compute_psnr = PeakSignalNoiseRatio(data_range=255.0, reduction='sum').to(device)
        compute_ssim = StructuralSimilarityIndexMeasure(data_range=255.0, reduction='sum').to(device)
        compute_lpips_alex = LearnedPerceptualImagePatchSimilarity(normalize=False, net_type='alex', reduction='sum').to(device)
        compute_lpips_vgg = LearnedPerceptualImagePatchSimilarity(normalize=False, net_type='vgg', reduction='sum').to(device)

    @torch.autocast(device_type="cuda", enabled=False)
    def run_detector(videos: torch.Tensor, resize_frames: tuple[int]=None, text: list[str]=None, videos_gt: torch.Tensor=None, waveform: torch.Tensor=None) -> torch.Tensor: # pylint: disable=unused-argument
        assert videos_gt is not None, 'videos_gt must be provided for reconstruction detector.'
        assert videos.shape == videos_gt.shape, f'Wrong shape: {videos.shape} vs {videos_gt.shape}'
        b, t, _c, h, w = videos.shape
        _t_new, h_new, w_new = (max(t, 16), h, w) if resize_frames is None else (max(t, 16), resize_frames[0], resize_frames[1])
        resize_frames = (h_new, w_new) if (h != h_new or w != w_new) else None # Not using t_new since we are extracting a single frame anyways.
        videos = rearrange_videos(videos, resize=resize_frames, temporal_detector=False) # [b * t, c, h, w]
        videos_gt = rearrange_videos(videos_gt.to(videos.device), resize=resize_frames, temporal_detector=False) # [b * t, c, h, w]
        videos_normalized = misc.maybe_rescale_uint8_to_fp32(videos) # [b * t, c, h, w]
        videos_gt_normalized = misc.maybe_rescale_uint8_to_fp32(videos_gt) # [b * t, c, h, w]

        # Compute L1, L2, PSNR and SSIM.
        l1 = (videos_normalized - videos_gt_normalized).abs().reshape(b, -1).mean(dim=1).unsqueeze(1) # [b, 1]
        l2 = (videos_normalized - videos_gt_normalized).pow(2).reshape(b, -1).mean(dim=1).unsqueeze(1).sqrt() # [b, 1]

        # Compute LPIPS.
        lpips_alex = run_metric_samplewise(compute_lpips_alex, videos_normalized, videos_gt_normalized).reshape(b, t).mean(dim=1).unsqueeze(1) # [b, 1]
        lpips_vgg = run_metric_samplewise(compute_lpips_vgg, videos_normalized, videos_gt_normalized).reshape(b, t).mean(dim=1).unsqueeze(1) # [b, 1]

        # Compute SSIM and PSNR.
        ssim = run_metric_samplewise(compute_ssim, videos.float(), videos_gt.float()).reshape(b, t).mean(dim=1).unsqueeze(1) # [b, 1]
        psnr = run_metric_samplewise(compute_psnr, videos.float(), videos_gt.float()).reshape(b, t).mean(dim=1).unsqueeze(1) # [b, 1]

        return torch.cat([l1, l2, lpips_alex, lpips_vgg, ssim, psnr], dim=1) # [b, 6]

    return Detector(run_detector, 'reconstruction')

def run_metric_samplewise(metric: Callable, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # We need to do this for stupid metric implementations that do not support no-reduction.
    return torch.stack([metric(x_i.unsqueeze(0), y_i.unsqueeze(0)).mean() for x_i, y_i in zip(x, y)])

#----------------------------------------------------------------------------
# Some data processing functions.

def rearrange_videos(videos: torch.Tensor, resize: int | tuple[int]=None, temporal_detector: bool=False, square_crop: bool=True) -> torch.Tensor:
    assert videos.ndim == 5 and videos.shape[2] in (1, 3), f"Wrong shape: {videos.shape}"
    if videos.shape[2] == 1:
        videos = videos.repeat([1, 1, 3, 1, 1]) # [b, t, c, h, w]
    if square_crop and videos.shape[3] != videos.shape[4]:
        videos = square_center_crop_videos(videos) # [b, t, c, h, w]
    if temporal_detector:
        videos = einops.rearrange(videos, 'b t c h w -> b c t h w').contiguous() # [batch_size, c, t, h, w]
    else:
        videos = videos.flatten(0, 1) # [b * t, c, h, w]
    if resize is not None:
        videos = torch.nn.functional.interpolate(misc.maybe_rescale_uint8_to_fp32(videos), size=resize, mode='area') # [b, t_r | null, c, h_r, w_r]
        videos = misc.maybe_rescale_fp32_to_uint8(videos) # [b, t_r | null, c, h_r, w_r]
    return videos


def square_center_crop_videos(videos: torch.Tensor, crop_size: int=None) -> torch.Tensor:
    assert len(videos.shape) == 5, f"Input tensor must be 5D: {videos.shape}"
    _, _, _, h, w = videos.shape
    crop_size = min(h, w) if crop_size is None else crop_size
    assert videos.shape[3] >= crop_size and videos.shape[4] >= crop_size, "Crop size should be less than or equal to videos dimensions"

    start_x = (w - crop_size) // 2
    start_y = (h - crop_size) // 2
    end_x = start_x + crop_size
    end_y = start_y + crop_size

    return videos[..., start_y:end_y, start_x:end_x]

#----------------------------------------------------------------------------
