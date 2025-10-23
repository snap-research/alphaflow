import os
from dataclasses import dataclass
from typing import Optional, Any, Iterator

from beartype import beartype
import torch

from src.utils import misc, distributed as dist
from src.data.datasets import FolderDataset
from src.data.utils import infer_batch_info, inf_loop_dataloader
from src.structs import BatchInfo, EasyDict, TensorGroup

#----------------------------------------------------------------------------

@dataclass(frozen=True)
class Data:
    dataset: torch.utils.data.Dataset
    dataset_val: Optional[torch.utils.data.Dataset | None]
    dataset_eval: Optional[torch.utils.data.Dataset | None]
    dataloader: torch.utils.data.dataloader
    iterator: Iterator
    batch_info: BatchInfo

    @staticmethod
    @beartype
    def filter_stream_in_cfg_(dataset_cfg: EasyDict, selected_stream: str=None):
        """Given an experiment config `experiment_cfg`, alters its `dataset_cfg` to contain jsut a single `selected_stream` stream."""
        if selected_stream is None:
            return

        dataset_cfg.streams = [s for s in dataset_cfg.streams if s.name == selected_stream] if dataset_cfg.streams is not None else None
        dataset_cfg.streams[0].ratio = 1.0

    @staticmethod
    def init_from_cfg(dataset_cfg: EasyDict) -> 'Data':
        dist.loginfo0('Loading dataset... ')

        if dataset_cfg.local_dir is not None:
            os.makedirs(os.readlink(dataset_cfg.local_dir) if os.path.islink(dataset_cfg.local_dir) else dataset_cfg.local_dir, exist_ok=True) # Create the dataset directory if it does not exist.

        dataloader, dataset_train, dataset_val, dataset_eval, batch_info = init_folder_dataset_kwargs(dataset_cfg)

        iterator = iter(inf_loop_dataloader(dataloader))

        return Data(
            dataset=dataset_train,
            dataset_val=dataset_val,
            dataset_eval=dataset_eval,
            dataloader=dataloader,
            iterator=iterator,
            batch_info=batch_info,
        )

def init_folder_dataset_kwargs(dataset_cfg) -> tuple[torch.utils.data.DataLoader, torch.utils.data.Dataset, torch.utils.data.Dataset | None, torch.utils.data.Dataset | None]:
    assert os.path.exists(dataset_cfg.src) or dataset_cfg.src.startswith('s3://'), f"Data path {dataset_cfg.src} is not an S3 url and does not exist."

    dataset_train_kwargs = EasyDict(
        src=dataset_cfg.src,
        resolution=dataset_cfg.resolution,
        data_type=dataset_cfg.data_type,
        max_size=dataset_cfg.max_size,
        metadata_ext=dataset_cfg.metadata_ext,
        label_shape=dataset_cfg.label_shape,
        shuffle_seed=dataset_cfg.shuffle_seed if dataset_cfg.shuffle else None,
        replace_broken_samples_with_random=dataset_cfg.replace_broken_samples_with_random,

        # Video-specific options.
        allow_shorter_videos=dataset_cfg.allow_shorter_videos,
        framerate=dataset_cfg.framerate,
        thread_type=dataset_cfg.thread_type,
        random_offset=dataset_cfg.random_offset,

        # Some S3-specific options.
        cache_dir=dataset_cfg.local_dir,
        s3_listdir_cache_dir=dataset_cfg.s3_listdir_cache_dir,

        # Debugging params.
        print_exceptions=dataset_cfg.print_exceptions,
        print_traceback=dataset_cfg.print_traceback,
    )

    dataset_val_kwargs = {**dataset_train_kwargs, **{'src': dataset_cfg.src_val}} if dataset_cfg.src_val is not None else None
    dataset_train = FolderDataset(**dataset_train_kwargs)
    dataset_val = FolderDataset(**dataset_val_kwargs) if dataset_val_kwargs is not None else None
    batch_info = infer_batch_info(dataset_cfg.batch_size, dataset_cfg.batch_gpu, dataset_cfg.num_accum_rounds)
    dataloader = torch.utils.data.DataLoader(
        dataset=get_dp_data_subset(dataset_train),
        batch_size=batch_info.batch_gpu,
        pin_memory=True,
        num_workers=dataset_cfg.num_workers,
        prefetch_factor=dataset_cfg.prefetch_factor,
        drop_last=dataset_cfg.drop_last,
        persistent_workers=dataset_cfg.persistent_workers,
        shuffle=dataset_cfg.shuffle,
    )

    if dataset_cfg.eval_stream is None:
        eval_dataset = dataset_val if dataset_cfg.use_val_data_for_eval_stream else dataset_train
    else:
        dataset_eval_kwargs = {**dataset_train_kwargs, **{'src': dataset_cfg.eval_stream.src, 'data_type': dataset_cfg.eval_stream.data_type}}
        eval_dataset = FolderDataset(**dataset_eval_kwargs)

    return dataloader, dataset_train, dataset_val, eval_dataset, batch_info

#----------------------------------------------------------------------------
# DataLoading utils.

def get_dp_data_subset(dataset: torch.utils.data.Dataset) -> torch.utils.data.Subset:
    """Returns a subset of items for the current data parallel rank."""
    assert isinstance(dataset, torch.utils.data.Dataset), f"Expected a torch Dataset, but got {type(dataset)}"
    idx = [(i * dist.get_world_size() + dist.get_rank()) % len(dataset) for i in range((len(dataset) - 1) // dist.get_world_size() + 1)]
    return torch.utils.data.Subset(dataset, idx)

def construct_full_cond_from_batch(batch: dict[str, Any], use_x_cond: bool=False) -> TensorGroup:
    cond = TensorGroup({k: torch.as_tensor(batch[k]) for k in ['label', 'dataset_id', 'framerate', 'resolution', 'audio'] if k in batch})
    if use_x_cond:
        cond.x_cond = misc.maybe_rescale_uint8_to_fp32(torch.as_tensor(batch['video'])) # [b, t, c, h, w]
    return cond

def construct_inputs_from_batch(batch: dict[str, Any], use_x_cond: bool=False, device: str | torch.device='cpu') -> tuple[torch.Tensor | TensorGroup, TensorGroup]:
    x = misc.maybe_rescale_uint8_to_fp32(batch['video'].to(device, non_blocking=True)) # [b, t, h, w, c]
    cond = construct_full_cond_from_batch(batch, use_x_cond).to(device, non_blocking=True) # [b, ...]
    return x, cond

#----------------------------------------------------------------------------
# Random utils.

def list_to_dict(values: list[tuple[Any, Any]]) -> dict[Any, Any]:
    """
    Converts a list of tuples to a dictionary.
    """
    return {tuple(k) if isinstance(k, list) else k: v for k, v in values}

#----------------------------------------------------------------------------
