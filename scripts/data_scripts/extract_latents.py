import os
import math
import json
import pickle
import traceback
import shutil

import torch
from omegaconf import DictConfig

from src.utils.os_utils import lean_tqdm, construct_class_by_name
from src.utils import misc
from src.data import Data, get_dp_data_subset
from src.structs import EasyDict
from src.training.network_utils import instantiate_network_kwargs
from infra.utils import recursive_instantiate
from infra.launch import launch_from_registry
from scripts.utils import slow_check_if_s3_path_exists
import src.utils.distributed as dist

#----------------------------------------------------------------------------

def extract_latents(cfg: DictConfig):
    recursive_instantiate(cfg)
    cfg = EasyDict.init_recursively(cfg)
    dist.init(timeout_seconds=3600 * 72) # Allow for 3 days of processing. If it takes longer, then it's probably not worth it.
    dist.init_random_state_and_cuda(cfg.training.seed, cfg.training.cudnn_benchmark, allow_tf32=cfg.training.allow_tf32)
    device = torch.device('cuda')
    net = data = None
    torch.set_grad_enabled(False)

    data = Data.init_from_cfg(cfg.dataset)
    datasets = {'val': data.dataset_val, 'train': data.dataset}

    for subset_name, dataset in datasets.items():
        if cfg.subset is not None and subset_name != cfg.subset:
            continue
        assert datasets[subset_name].epoch_size < 1_500_000, f"Dataset epoch_size={datasets[subset_name].epoch_size} is too big to process. Maybe we should extract the latents on the fly?"
        completion_flag_file = os.path.join(cfg.dst, f'{subset_name}.done')

        if slow_check_if_s3_path_exists(completion_flag_file):
            dist.loginfo0(f"Latents for subset '{subset_name}' already exist. Skipping...")
            continue

        if net is None:
            # Delay the network instantiation until we know that we need to do some work.
            interface_kwargs = dict(resolution=cfg.dataset.resolution, img_channels=3, label_shape=cfg.dataset.label_shape)
            network_kwargs, _ = instantiate_network_kwargs(cfg)
            ldm = construct_class_by_name(**network_kwargs, **interface_kwargs) # subclass of torch.nn.Module
            net = ldm.autoencoder.eval().requires_grad_(False).to(device)

        write_dataloader(
            cfg,
            subset_name,
            dataset,
            net=net,
            device=device,
        )

        # Setting a barrier for all processes to finish writing before exiting.
        dist.barrier()
        dist.loginfo0(f"All processes have finished writing the data for subset_name '{subset_name}'.")

        if dist.get_rank() == 0:
            completion_flag_file_local = os.path.join(cfg.tmp_dir, f'{subset_name}.done')
            with open(completion_flag_file_local, 'w') as f:
                f.write('done\n')
            upload_and_delete((completion_flag_file_local, completion_flag_file))
            dist.loginfo0(f"Saved completion flag file for subset_name '{subset_name}' to {completion_flag_file}.")

def write_dataloader(cfg, subset_name: str, dataset, **kwargs):
    dist.loginfo0(f'Dataset shape: {[dataset.epoch_size, *cfg.dataset.resolution]}')
    dst = os.path.join(cfg.dst, subset_name)

    dataloader = torch.utils.data.DataLoader(
        get_dp_data_subset(dataset) if isinstance(dataset, torch.utils.data.Dataset) else dataset,
        batch_size=cfg.dataset.batch_gpu,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=cfg.dataset.prefetch_factor,
    )

    approx_num_batches_to_process = math.ceil(len(dataset) / cfg.dataset.batch_gpu / dist.get_world_size())

    for batch_idx, batch in lean_tqdm(enumerate(dataloader), total=approx_num_batches_to_process, desc=f'Extracting latents for {subset_name} into {dst}', disable=not dist.is_main_process()):
        try:
            # if batch_idx >= 16: break # For debugging
            write_batch(cfg.tmp_dir, dst, batch, save_metadata=cfg.save_metadata, **kwargs)
        except Exception as e: # pylint: disable=broad-except
            print(f'Error processing batch: {e}')
            traceback.print_exc()
            continue

def write_batch(tmp_dir, dst: str, batch, net, device, save_metadata: bool=False):
    video = misc.maybe_rescale_uint8_to_fp32(batch['video'].to(device)) # [b, t, c, h, w]
    _, ctx = net(video.to(device), encode=True, decode=False, return_extra_output=True)
    mean, logvar = ctx['latents_dist_mean'], ctx['latents_dist_logvar']

    if mean.ndim == 5:
        # Squeeze the time dimension if it's an image, so we have [b, c, h, w].
        mean = mean.squeeze(1)
        logvar = logvar.squeeze(1)

    for sample_idx in range(video.shape[0]):
        latents_dict = {
            'mean': mean[sample_idx].cpu().float().numpy(), # [t | null, c, h, w]
            'logvar': logvar[sample_idx].cpu().float().numpy(), # [t | null, c, h, w]
            'input_shape': list(video.shape[1:]), # [t, c, h, w]
        }
        sample_key = batch['__sample_key__'][sample_idx]
        filename = f"{sample_key}.latents.pkl"
        dst_path = save_upload_delete(tmp_dir, dst, latents_dict, filename)

        cur_metadata = {
            'latents_url': dst_path,
            'input_shape': list(video.shape[1:]),
        }

        if save_metadata:
            class_label = batch['label'][sample_idx].cpu() # [num_classes]
            cur_metadata['class'] = int(class_label.argmax().item())
            metadata_filename = f"{sample_key}.latents.meta.json"
            save_upload_delete(tmp_dir, dst, cur_metadata, metadata_filename, storage='json')


def save_upload_delete(tmp_dir, dst: str, obj: dict, filename: str, storage: str='pickle') -> str:
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, filename)
    dst_path = os.path.join(dst, filename)
    if not dst.startswith("s3://"):
        os.makedirs(dst, exist_ok=True)
    if storage == 'pickle':
        with open(tmp_path, 'wb') as f:
            pickle.dump(obj, f)
    elif storage == 'json':
        with open(tmp_path, 'w') as f:
            json.dump(obj, f)
    else:
        raise ValueError(f'Unsupported storage format: {storage}')

    upload_and_delete((tmp_path, dst_path))

    return dst_path


def upload_and_delete(task_input):
    tmp_path, dst_path = task_input
    shutil.move(tmp_path, dst_path)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    launch_from_registry(extract_latents, config_path='../configs/extract_latents.yaml')

#----------------------------------------------------------------------------
