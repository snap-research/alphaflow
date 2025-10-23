import os
import uuid
import pickle
from typing import Self

import numpy as np
import torch

from src.utils import distributed as dist
from src.structs import BasicStats, EasyDict

#----------------------------------------------------------------------------

class FeatureStats:
    def __init__(self, capture_all: bool=False, capture_mean_cov: bool=False, capture_basic_stats: bool=False, max_items=None):
        self.capture_all = capture_all
        self.capture_mean_cov = capture_mean_cov
        self.capture_basic_stats = capture_basic_stats
        self.max_items = max_items
        self.num_items = 0
        self.feature_shape = None
        self.all_features = None
        self.raw_mean = None
        self.raw_cov = None
        self.raw_var = None
        self.raw_min = None
        self.raw_max = None
        self._ranks_done = torch.tensor([False] * dist.get_world_size(), dtype=torch.bool, device='cuda') # [world_size]

    def set_feature_shape(self, feature_shape: tuple[int]):
        assert isinstance(feature_shape, tuple), f'Expected feature_shape to be a tuple, but got {type(feature_shape)} instead.'
        if self.feature_shape is not None:
            assert feature_shape == self.feature_shape, f'Expected feature_shape to be {self.feature_shape}, but got {feature_shape} instead.'
        else:
            self.feature_shape = feature_shape
            self.all_features = []
            if self.capture_mean_cov or self.capture_basic_stats:
                self.raw_mean = np.zeros(self.feature_shape, dtype=np.float64) # [d_0, ..., d_n]
            if self.capture_basic_stats:
                self.raw_var = np.zeros(self.feature_shape, dtype=np.float64) # [d_0, ..., d_n]
                self.raw_min = np.full(self.feature_shape, np.inf, dtype=np.float64) # [d_0, ..., d_n]
                self.raw_max = np.full(self.feature_shape, -np.inf, dtype=np.float64) # [d_0, ..., d_n]
            if self.capture_mean_cov:
                assert len(self.feature_shape) == 1, f'To capture covariance, expected feature_shape to have 1 dimension, but got {len(self.feature_shape)} instead.'
                self.raw_cov = np.zeros([self.feature_shape[0], self.feature_shape[0]], dtype=np.float64) # [d, d]

    def is_full(self):
        return (self.max_items is not None) and (self.num_items >= self.max_items)

    def append(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float32) # [n, d]
        assert x.ndim >= 2, f'Expected x to have at least 2 dimensions, but got {x.ndim} instead.'

        # Cut off the extra items.
        if (self.max_items is not None) and (self.num_items + x.shape[0] > self.max_items):
            if self.num_items >= self.max_items:
                return # Nothing to add
            x = x[:self.max_items - self.num_items]

        self.set_feature_shape(x.shape[1:])
        self.num_items += x.shape[0]
        if self.capture_all:
            self.all_features.append(x)
        if self.capture_mean_cov or self.capture_basic_stats:
            x64 = x.astype(np.float64) # [n, d]
            self.raw_mean += x64.sum(axis=0) # [d]
        if self.capture_basic_stats:
            self.raw_var += (x64 ** 2).sum(axis=0) # [d]
            self.raw_min = np.minimum(self.raw_min, x64.min(axis=0)) # [d]
            self.raw_max = np.maximum(self.raw_max, x64.max(axis=0)) # [d]
        if self.capture_mean_cov:
            self.raw_cov += x64.T @ x64

    def append_torch(self, x):
        assert isinstance(x, torch.Tensor) and x.ndim >= 2, f'Expected x to be a tensor with at least 2 dimensions, but got {x.ndim} instead.'
        x = dist.gather_concat_not_nan(x.flatten(start_dim=1)).reshape(-1, *x.shape[1:]) # [world_size * n, d]
        self.append(x.float().cpu().numpy())

    def get_all(self):
        assert self.capture_all
        return np.concatenate(self.all_features, axis=0)

    def get_all_torch(self):
        return torch.from_numpy(self.get_all())

    def get_mean_cov(self):
        assert self.capture_mean_cov
        mean = self.raw_mean / self.num_items
        cov = self.raw_cov / self.num_items
        cov = cov - np.outer(mean, mean)
        return mean, cov

    def get_basic_stats(self) -> BasicStats:
        assert self.capture_basic_stats
        feats_mean = self.raw_mean / self.num_items # [d]
        feats_std = np.sqrt(self.raw_var / self.num_items - feats_mean ** 2 + 1e-8) # [d]
        feats_min = self.raw_min # [d]
        feats_max = self.raw_max # [d]
        return BasicStats(mean=feats_mean, std=feats_std, min=feats_min, max=feats_max, num_samples_used=self.num_items)

    def set_rank_done(self, rank: int, done: bool):
        self._ranks_done[rank] = done # [1]
        if dist.get_world_size() > 1:
            torch.distributed.all_reduce(self._ranks_done, op=torch.distributed.ReduceOp.MAX) # [world_size]

    def are_all_ranks_done(self) -> bool:
        return self._ranks_done.all()

    def safe_save(self, cache_file: os.PathLike):
        if cache_file is None or not dist.is_main_process():
            return

        # Save to cache.
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        temp_file = cache_file + '.' + uuid.uuid4().hex
        self.save(temp_file)
        os.replace(temp_file, cache_file) # atomic

    def safe_save_partial(self, cache_file: os.PathLike):
        cache_file_partial = cache_file.replace('.pkl', f'.partial-{self.num_items:011d}.pkl')
        self.safe_save(cache_file_partial)
        dist.loginfo0(f'Saved {self.num_items} items to {cache_file_partial}.')

    def save(self, pkl_file):
        with open(pkl_file, 'wb') as f:
            pickle.dump(self.__dict__, f)

    @staticmethod
    def load(pkl_file):
        with open(pkl_file, 'rb') as f:
            state = EasyDict(pickle.load(f))
        return FeatureStats.load_from_state(state)

    @staticmethod
    def load_from_state(state: dict | Self) :
        stats = FeatureStats(capture_all=state.capture_all, max_items=state.max_items)
        stats.__dict__.update(state if isinstance(state, dict) else state.__dict__)
        return stats

#----------------------------------------------------------------------------
