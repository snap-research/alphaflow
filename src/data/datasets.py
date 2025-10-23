import os
import json
import pickle
from io import BytesIO
from typing import Any
import traceback
import zipfile

from beartype import beartype
import numpy as np
import torch
from PIL import Image

from src.data.utils import decode_video, convert_pil_image_to_byte_tensor, sample_image_vae_latents, sample_video_vae_latents, lean_resize_frames, path_key, copy_file, get_s3_file_paths, VIDEO_EXTENSIONS
from src.utils.os_utils import file_ext
from src.structs import DataSampleType as DSType
import src.utils.distributed as dist

#----------------------------------------------------------------------------

class FolderDataset(torch.utils.data.Dataset):
    _SRC_TYPES = ['dir', 'zip', 's3']

    def __init__(self,
        src: str,
        max_size: int | None               = None,       # Artificially limit the size of the dataset. None = no limit.
        label_shape: tuple[int] = (),                    # Shape of the label. Empty tuple = no label.
        shuffle_seed: int | None           = 0,          # Random seed for shuffling the data idx. None = no shuffling.
        data_type: DSType | str  = False,                # Dataset contains precomputed latents instead of images.
        metadata_ext = '.json',
        cache_dir: str | None=None,                             # Copy files from src to the specified local cache directory while iterating.
        s3_listdir_cache_dir: str=None,                  # When using S3 paths, cache the list of files to this local file.
        resolution: tuple[int, int]=None,                # Desired resolution. None = keep original resolution.
        replace_broken_samples_with_random: bool=False,  # If True, replaces broken samples with random samples from the dataset.

        # Video-specific params.
        thread_type: str='NONE',                         # Thread type to use for decoding videos.
        framerate: float=None,                           # Desired framerate. None = use original framerate for each video.
        allow_shorter_videos: bool=False,                # Should we allow videos that are shorter than `load_n_consecutive`?
        random_offset: bool=True,                        # Should we use a random offset when extracting frames from videos?
        frame_seek_timeout_sec: float=10.0,              # Timeout in seconds for seeking to a specific frame in a video.

        # Debugging params.
        print_exceptions: bool   = True,
        print_traceback: bool    = True,                 # Should we print the traceback when an exception occurs?
    ):
        self._src = src
        self.name = path_key(src)
        self.data_type = DSType.from_str(data_type) if isinstance(data_type, str) else data_type
        self._shuffle_seed = shuffle_seed
        self._label_shape = label_shape
        self._max_size = max_size
        self._zipfile = None
        self._print_exceptions = print_exceptions
        self._print_traceback = print_traceback
        self.resolution = resolution
        self._metadata_ext = metadata_ext
        self._cache_dir = cache_dir
        self._s3_listdir_cache_dir = s3_listdir_cache_dir
        self._replace_broken_samples_with_random = replace_broken_samples_with_random

        # Video-specific params.
        self._thread_type = thread_type
        self._framerate = framerate
        self._allow_shorter_videos = allow_shorter_videos
        self._random_offset = random_offset
        self._frame_seek_timeout_sec = frame_seek_timeout_sec

        # Initializing the index.
        self._all_fnames: set[str] = set()
        self._data_fnames: list[str] = []
        self._init_all_fnames()
        self._init_data_fnames()

        # Apply max_size.
        if max_size is not None and len(self._data_fnames) > max_size:
            assert self._shuffle_seed is not None, 'When using max_size, shuffle_seed must be specified to get a random subset of the data.'
            self._data_fnames = self._data_fnames[:max_size]

        self.epoch_size = len(self._data_fnames)

    def _init_all_fnames(self):
        if self._src.startswith('s3://'):
            assert self._cache_dir is not None, 'When using S3 paths, `cache_dir` must be specified to cache the files locally.'
            assert self._src.endswith('/'), f'When using S3 paths, the path must end with a slash (e.g. s3://bucket-name/dataset/), but got: {self._src}'
            self._type = 's3'
            s3_contents: list[str] = get_s3_file_paths(self._src, is_main_process=dist.get_rank() == 0, s3_listdir_cache_dir=self._s3_listdir_cache_dir)
            s3_contents = [path[len(self._src):] for path in s3_contents if not path.endswith('/')] # Make paths relative to the S3 "directory".
            self._all_fnames = set(s3_contents)
        elif os.path.isdir(self._src):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._src) for root, _dirs, files in os.walk(self._src) for fname in files}
        elif file_ext(self._src) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError(f'Path must point to a directory or zip, but got: {self._src}')

    def _init_data_fnames(self):
        Image.init()
        data_ext = {DSType.VIDEO: VIDEO_EXTENSIONS, DSType.IMAGE: Image.EXTENSION, DSType.VIDEO_LATENT: {'.pkl'}, DSType.IMAGE_LATENT: {'.pkl'}}[self.data_type]
        self._data_fnames = sorted(fname for fname in self._all_fnames if file_ext(fname) in data_ext) # [num_videos]
        if self._shuffle_seed is not None:
            np.random.RandomState(self._shuffle_seed % (1 << 31)).shuffle(self._data_fnames)
        assert len(self._data_fnames) > 0, f'No data files found in the specified path: {self._src}.'

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._src)
        return self._zipfile

    def _load_file(self, fname: str, mode: str | None=None):
        if self._type == 'dir':
            src_path = os.path.join(self._src, fname)
            if self._cache_dir is not None:
                # Sometimes, we want to move the files from some slow filesystem to a local node disk.
                dst_path = os.path.join(self._cache_dir, self.name, fname)
                copy_file(src_path, dst_path, skip_if_exists=True)
            else:
                dst_path = src_path
            return open(dst_path, mode or 'rb')
        elif self._type == 'zip':
            return self._get_zipfile().open(fname, mode or 'r')
        elif self._type == 's3':
            # Downloads the file from S3 to local cache_dir and opens it.
            dst_path = os.path.join(self._cache_dir, self.name, fname)
            copy_file(os.path.join(self._src, fname), dst_path, skip_if_exists=True)
            return open(dst_path, mode or 'rb')
        else:
            raise RuntimeError(f'Invalid dataset type: {self._type}')

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(self.__dict__, _zipfile=None)

    def __del__(self):
        try:
            self.close()
        except: # pylint: disable=bare-except
            pass

    def __len__(self):
        return len(self._data_fnames)

    @beartype
    def set_progress(self, epoch: int, sample_in_epoch: int | None=None) -> None:
        pass

    @beartype
    def load_metadata(self, idx) -> tuple[dict[str, Any] | None, np.ndarray | int]:
        if len(self._label_shape) == 0 and self._metadata_ext is None:
            return None, np.array(-1)

        metadata = None
        if self._metadata_ext is not None:
            metadata_path = os.path.splitext(self._data_fnames[idx])[0] + self._metadata_ext
            assert metadata_path in self._all_fnames, f'Metadata file not found: {metadata_path}'
            with self._load_file(metadata_path, 'r') as f:
                metadata = json.load(f)

        if len(self._label_shape) == 0:
            label = -1
        else:
            assert metadata is not None and 'class' in metadata, f'No "class" field found in the metadata: {metadata_path}'
            label = np.array(metadata['class'])
            if label.dtype == np.int64:
                assert len(self._label_shape) == 1, f'Cannot convert integer label to one-hot with shape {self._label_shape}'
                onehot = np.zeros(self._label_shape, dtype=np.float32)
                onehot[label] = 1
                label = onehot

        return metadata, label

    def __getitem__(self, idx: int, num_retries: int = 10) -> dict:
        assert 0 <= idx < len(self), f'Index out of range: {idx}'
        cur_idx = idx
        for _ in range(num_retries):
            try:
                return self.__unsafe_getitem__(cur_idx)
            except Exception as e: # pylint: disable=broad-except
                if self._print_exceptions:
                    print(f"Exception in __getitem__({cur_idx}): {e}")
                if self._print_traceback:
                    traceback.print_exc()
                if self._replace_broken_samples_with_random:
                    cur_idx = np.random.randint(len(self))
                continue
        raise RuntimeError(f'Failed to read item {idx} after {num_retries} retries.')

    def __unsafe_getitem__(self, idx: int) -> dict:
        sample_file = self._load_file(self._data_fnames[idx], 'rb')
        sample_metadata, label = self.load_metadata(idx)
        frames = decode_frames_from_sample(
            sample=sample_file,
            data_type=self.data_type,
            sample_metadata=sample_metadata,
            random_offset=self._random_offset,
            frame_seek_timeout_sec=self._frame_seek_timeout_sec,
            allow_shorter_videos=self._allow_shorter_videos,
            framerate=self._framerate,
            resolution=self.resolution,
        ) # [t, c, h, w]

        assert isinstance(frames, torch.Tensor), f"frames is not a torch.Tensor: {type(frames)}"

        return dict(
            video=frames,
            label=label,
            __sample_key__=os.path.splitext(os.path.basename(self._data_fnames[idx]))[0].replace('/', '_'),
        )

    def get_identifier_desc(self) -> str:
        res_str = "x".join([str(r) for r in self.resolution]) if self.resolution is not None else "orig"
        return f'{self.name}-ms{self._max_size}-r{res_str}-{str(self.data_type).lower()}-fr{self._framerate}-asv{self._allow_shorter_videos}-off{int(self._random_offset)}'

#----------------------------------------------------------------------------
# Some data loading utils.

def decode_frames_from_sample(
    sample: BytesIO,
    data_type: DSType,
    sample_metadata: dict[str, Any] | None,
    random_offset: bool = True,
    frame_seek_timeout_sec: float = 10.0,
    allow_shorter_videos: bool = False,
    framerate: float = None,
    resolution: tuple[int, int, int] = None,
) -> torch.Tensor:
    if data_type == DSType.VIDEO:
        frames = decode_video(
            video_file=sample,
            num_frames_to_extract=resolution[0],
            random_offset=random_offset,
            frame_seek_timeout_sec=frame_seek_timeout_sec,
            allow_shorter_videos=allow_shorter_videos,
            framerate=framerate,
            thread_type='NONE',
        )[0] # (t, Image)
        resize_to = tuple(resolution[1:])
    elif data_type == DSType.IMAGE:
        frames = [Image.open(sample)] # (1, Image)
        resize_to = tuple(resolution[1:])
    elif data_type == DSType.IMAGE_LATENT:
        frames = sample_image_vae_latents(pickle.load(sample)).unsqueeze(0) # [1, lc, lh, lw]
        resize_to = None
    elif data_type == DSType.VIDEO_LATENT:
        assert sample_metadata is not None, "sample_metadata must be provided for VIDEO_LATENT data_type."
        frames = sample_video_vae_latents(
            pickle.load(sample),
            orig_shape=tuple(sample_metadata['input_shape']),
            num_rgb_frames_to_extract=resolution[0],
            fps_orig=sample_metadata.get('framerate', None),
            fps_trg=framerate,
            random_offset=random_offset,
        ) # [lt, lc, lh, lw]
        resize_to = None
    else:
        raise NotImplementedError(f"Unknown data type: {data_type}")

    frames = lean_resize_frames(frames, resize_to) if resize_to is not None else frames # (t, Image)
    frames = torch.stack([convert_pil_image_to_byte_tensor(f, cut_alpha=True) for f in frames]) if isinstance(frames[0], Image.Image) else frames # [t, c, h, w]

    return frames

#----------------------------------------------------------------------------
