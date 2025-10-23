import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch_dct import dct_2d, idct_2d
import einops

from src.structs import TokenType
from src.structs import EasyDict, TensorGroup
from src.utils import misc

#----------------------------------------------------------------------------

def maybe_pad_videos(x: torch.Tensor, padding: tuple[int, int, int]=None, mode: str='constant') -> torch.Tensor:
    """Assumes that the input is of shape [n, t, c, h, w]."""
    if not padding is None:
        # Please note that `F.pad`'s input format is very weird. Our padding format is (t_pad, h_pad, w_pad).
        pad_t, pad_h, pad_w = padding
        if mode == 'constant':
            x = F.pad(x, [pad_w, pad_w, pad_h, pad_h, 0, 0, pad_t, pad_t, 0, 0], mode='constant', value=0.0) # [n, t_padded, c, h_padded, w_padded] pylint: disable=not-callable
        elif mode == 'replicate':
            assert pad_t == 0, f'Replicating in time is not supported (would require shapings): {pad_t}'
            x = F.pad(x, [pad_w, pad_w, pad_h, pad_h, 0, 0], mode='replicate') # [n, t_padded, c, h_padded, w_padded] pylint: disable=not-callable
    return x


def maybe_unpad_videos(x: torch.Tensor, padding: tuple[int, int, int]=None) -> torch.Tensor:
    """Assumes that the input is of shape [n, t, c, h, w]."""
    if not padding is None:
        pad_t, pad_h, pad_w = padding
        _b, t, _c, h, w = x.shape
        x = x[:, pad_t:t - pad_t, :, pad_h:h - pad_h, pad_w:w - pad_w] # [n, t, c, h, w]
    return x


def sample_frames_masks(videos: torch.Tensor, mask_sampling_cfg: EasyDict, is_video_mask: torch.Tensor=None) -> torch.Tensor:
    # For each element in the batch, we sample a random mask for it, given the probabilities.
    is_video_mask = is_video_mask.bool().cpu() if is_video_mask is not None else torch.ones(videos.shape[0]).bool() # [b]
    assert is_video_mask.shape == (videos.shape[0],), f'Invalid shape for the video mask: {is_video_mask.shape}'
    masks = []
    for is_video in is_video_mask:
        mask_types, mask_probs = zip(*(mask_sampling_cfg.video_masks if is_video else mask_sampling_cfg.image_masks).items())
        assert sum(mask_probs) == 1.0, f'Mask probabilities should sum to 1: {mask_probs}'
        mask_type = np.random.choice(mask_types, size=1, p=mask_probs)
        masks.append([t.value for t in create_frames_mask_by_type(videos.shape[1], mask_type)])
    masks = torch.tensor(masks, device=videos.device, dtype=torch.int64) # [b, t]
    return masks


def create_frames_mask_by_type(num_frames: int, mask_type: str) -> list[TokenType]:
    assert num_frames % 2 == 0, f'Number of frames should be even: {num_frames}'

    if mask_type == 'random_frame_enc': # [M, M, Q, M, M, M, M, M]
        mask = [TokenType.MISSING] * num_frames
        mask[np.random.randint(num_frames)] = TokenType.QUERY
    elif mask_type == 'random_frame_cond': # [C, C, Q, C, C, C, C, C]
        mask = [TokenType.CONTEXT] * num_frames
        mask[np.random.randint(num_frames)] = TokenType.QUERY
    elif mask_type == 'first_frame_cond': # [C, Q, Q, Q, Q, Q, Q, Q]
        return [TokenType.CONTEXT] + [TokenType.QUERY] * (num_frames - 1)
    elif mask_type == 'last_frame_cond': # [Q, Q, Q, Q, Q, Q, Q, C]
        return [TokenType.QUERY] * (num_frames - 1) + [TokenType.CONTEXT]
    elif mask_type == 'first_chunk': # [M, M, M, M, Q, Q, Q, Q]
        return [TokenType.MISSING] * (num_frames // 2) + [TokenType.QUERY] * (num_frames // 2)
    elif mask_type == 'second_chunk': # [C, C, C, C, Q, Q, Q, Q]
        return [TokenType.CONTEXT] * (num_frames // 2) + [TokenType.QUERY] * (num_frames // 2)
    elif mask_type == 'last_frame_enc': # [M, M, M, M, M, M, M, Q]
        return [TokenType.MISSING] * (num_frames - 1) + [TokenType.QUERY]
    elif mask_type == 'first_frame_enc': # [Q, M, M, M, M, M, M, M]
        return [TokenType.QUERY] + [TokenType.MISSING] * (num_frames - 1)
    elif mask_type == 'first_chunk_cond': # [C, C, C, C, Q, Q, Q, Q]
        return [TokenType.CONTEXT] * (num_frames // 2) + [TokenType.QUERY] * (num_frames // 2)
    elif mask_type == 'border_frames_cond': # [C, Q, Q, Q, Q, Q, Q, C]
        return [TokenType.CONTEXT] + [TokenType.QUERY] * (num_frames - 2) + [TokenType.CONTEXT]
    elif mask_type == 'full_enc': # [Q, Q, Q, Q, Q, Q, Q, Q]
        return [TokenType.QUERY] * num_frames
    else:
        raise NotImplementedError(f'Invalid mask type: {mask_type}')

    return mask


def create_autoregressive_masks(num_frames_per_clip: int, total_video_len: int, overlap_size: int, batch_size: int, is_first_frame_conditioned: bool=False, device: torch.device='cpu') -> torch.Tensor:
    # The first clip is fully query-based, the subsequent ones have the previous one as a context.
    # If we condition on the very first frame, then it goes as a context in the first clip.
    assert total_video_len >= num_frames_per_clip, f'Total video length should be greater than the number of frames per clip: {total_video_len} vs {num_frames_per_clip}'
    assert overlap_size < num_frames_per_clip, f'Overlap size should be smaller than the number of frames per clip: {overlap_size} vs {num_frames_per_clip}'
    assert (total_video_len - num_frames_per_clip) % (num_frames_per_clip - overlap_size) == 0, f'Overlap size should be such that the total video length can be divided by the number of frames per clip: {total_video_len} vs {num_frames_per_clip}'

    # First clip + the remaining clips.
    num_clips = 1 + (total_video_len - num_frames_per_clip) // (num_frames_per_clip - overlap_size) # [1]

    # Creating the masks sequence for a single video, and then repeating it for the batch.
    masks = []
    for clip_idx in range(num_clips):
        if clip_idx == 0:
            mask = [TokenType.QUERY] * num_frames_per_clip
            mask[0] = TokenType.CONTEXT if is_first_frame_conditioned else TokenType.QUERY
        else:
            mask = [TokenType.CONTEXT] * overlap_size + [TokenType.QUERY] * (num_frames_per_clip - overlap_size)
        masks.append(mask)
    masks = torch.tensor([[t.value for t in mask] for mask in masks], device=device, dtype=torch.int64) # [num_clips, num_frames_per_clip]
    masks = masks.unsqueeze(1).repeat(1, batch_size, 1) # [num_clips, b, num_frames_per_clip]

    return masks


def compute_annealed_weight(cur_step: int, start_weight: float, end_weight: float, annealing_steps: int, annealing_start_step: int=0) -> float:
    if cur_step < annealing_start_step:
        return start_weight
    elif annealing_start_step + annealing_steps <= cur_step:
        return end_weight
    else:
        # Linearly anneal the weight from the start to the end over the annealing steps.
        return start_weight + (end_weight - start_weight) * (cur_step - annealing_start_step) / annealing_steps


def compute_scheduled_weight(cur_step: int, schedule: list[tuple[int, float]]) -> float:
    assert len(schedule) > 0, f'Schedule should not be empty: {schedule}'
    assert [step_1 < step_2 for step_1, step_2 in zip(schedule[:-1], schedule[1:])], f'Steps should be in increasing order: {schedule}'
    for step, weight in schedule:
        if cur_step < step:
            return weight
    return schedule[-1][1]


def infer_is_vertical_flag(cond: Optional[TensorGroup]=None) -> bool:
    """
    Uses the conditioning to infer if the video is vertical and additionall performs a check on the input tensor.
    TODO: always provide `is_vertical` as a part of the conditioning.
    TODO: (eventually) replace `is_vertical` with the aspect ratio.
    """
    if cond is None or not 'is_vertical' in cond:
        is_vertical = False
    else:
        assert cond['is_vertical'].bool().all() or not cond['is_vertical'].bool().any(), f"All the samples must be the same: {cond['is_vertical']}."
        is_vertical = cond['is_vertical'].bool()[0].item()
    return is_vertical

#----------------------------------------------------------------------------

def compute_resolutions(input_resolution: tuple[int, int, int], num_blocks: int=None, min_resolution: int=None) -> list[tuple[int, int, int]]:
    assert num_blocks is None or min_resolution is None, f'Only one of the `num_blocks` or `min_resolution` should be provided: {num_blocks} vs {min_resolution}'
    num_blocks = math.floor(np.log2(max(input_resolution))) - int(np.log2(min_resolution)) + 1
    resolutions = [input_resolution]
    for level_idx in range(1, num_blocks):
        resolution_in = resolutions[-1]
        resolution_out = tuple((r // 2 if r // 2 >= 2 else r) for r in resolution_in)
        assert resolution_out != resolution_in, f"Resolution {resolution_out} is the same as the previous one {resolutions[-1]} (level_idx: {level_idx})"
        resolutions.append(tuple(resolution_out))
    return resolutions

#----------------------------------------------------------------------------
# Some DCT utils.

def cut_dct2d_high_freqs(x: torch.Tensor, block_size: int=None, zigzag: bool=False, keep_count: int=None, cut_ratio: float=None) -> torch.Tensor:
    assert keep_count is not None or cut_ratio is not None, f"Either keep_count or cut_ratio should be specified, but got {keep_count} and {cut_ratio}"
    assert not (keep_count is not None and cut_ratio is not None), f"Only one of keep_count or cut_ratio should be specified, but got {keep_count} and {cut_ratio}"
    h, w = x.shape[-2], x.shape[-1]
    block_size = block_size if block_size is not None else math.gcd(h, w) # [1]
    assert 0.0 <= cut_ratio <= 1.0, f"cut_ratio should be in [0, 1], but got {cut_ratio}"
    assert h >= block_size and w >= block_size, f"Height and width should be at least block_size, but got {h} and {w} and {block_size}"
    assert h % block_size == 0 and w % block_size == 0, f"Height and width should be divisible by block_size, but got {h} and {w} and {block_size}"

    x = einops.rearrange(x, '... (nbh bh) (nbw bw) -> ... nbh nbw bh bw', nbh=h // block_size, bh=block_size, nbw=w // block_size, bw=block_size)
    x_dct = dct_2d(x, norm='ortho') # [..., b, b]
    if zigzag:
        assert keep_count is None or 0 <= keep_count <= block_size * block_size, f"for zigzag order, keep_count should be in [0, block_size ** 2], but got {keep_count} and block_size ** 2={block_size**2}"
        keep_count = keep_count if keep_count is not None else int((1.0 - cut_ratio) * block_size * block_size) # [1]
        row_idx, col_idx = zip(*zigzag_indices(block_size)) # [b * b], [b * b]
        row_idx_zero = row_idx[keep_count:] # [b * b - keep_count]
        col_idx_zero = col_idx[keep_count:] # [b * b - keep_count]
        x_dct[..., row_idx_zero, col_idx_zero] = 0.0
    else:
        assert keep_count is None or 0 <= keep_count <= block_size, f"for block order, keep_count should be in [0, block_size], but got {keep_count} and {block_size}"
        keep_count = keep_count if keep_count is not None else int((1.0 - cut_ratio) * block_size) # [1]
        x_dct[..., keep_count:, :] = 0.0
        x_dct[..., :, keep_count:] = 0.0
    x_idct = idct_2d(x_dct, norm='ortho') # [..., b, b]
    x_idct = einops.rearrange(x_idct, '... nbh nbw bh bw -> ... (nbh bh) (nbw bw)', nbh=h // block_size, bh=block_size, nbw=w // block_size, bw=block_size) # [..., h, w]

    return x_idct

def reg_dc_dct2d_high_freqs(x: torch.Tensor, block_size: int=None, power: float=2) -> torch.Tensor:
    h, w = x.shape[-2], x.shape[-1]
    block_size = block_size if block_size is not None else math.gcd(h, w) # [1]
    x = einops.rearrange(x, '... (nbh bh) (nbw bw) -> ... nbh nbw bh bw', nbh=h // block_size, bh=block_size, nbw=w // block_size, bw=block_size)
    x_dct = dct_2d(x, norm='ortho') # [..., b, b]
    amplitude = x_dct.abs() # [..., b, b]
    x_coords = torch.linspace(0, 1, steps=block_size, device=x.device, dtype=x.dtype).unsqueeze(0).expand(block_size, -1) # [b, b]
    y_coords = torch.linspace(0, 1, steps=block_size, device=x.device, dtype=x.dtype).unsqueeze(1).expand(-1, block_size) # [b, b]
    weight = (x_coords + y_coords + 1e-16).float().pow(power) / 4.0 # [b, b]
    weight = misc.unsqueeze_left(weight, amplitude) # [..., b, b]
    loss = (amplitude * weight).reshape(amplitude.shape[0], -1).mean(dim=1) # [batch_size]

    return loss


def zigzag_indices(n: int):
    """
    Returns a list of (row, col) indices following the *standard JPEG* zigzag:
      - Even diagonals (sum = 0,2,4,...) go bottom-up
      - Odd diagonals (sum = 1,3,5,...) go top-down
    """
    indices = []
    for s in range(2 * n - 1):
        if s % 2 == 0:
            # even sum => bottom-up
            for i in range(s, -1, -1):
                j = s - i
                if 0 <= i < n and 0 <= j < n:
                    indices.append((i, j))
        else:
            # odd sum => top-down
            for i in range(s + 1):
                j = s - i
                if 0 <= i < n and 0 <= j < n:
                    indices.append((i, j))
    return indices

def zigzag_flatten(block: torch.Tensor | np.ndarray) -> list:
    assert block.ndim == 2, f"Wrong shape for the block: {block.shape}"
    idx = zigzag_indices(block.shape[0])
    return [block[r][c] for (r, c) in idx]

#----------------------------------------------------------------------------
