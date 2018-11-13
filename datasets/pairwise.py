from __future__ import absolute_import

import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from utils.bbox_utils import to_percentage_coords


class PairSampler(Dataset):

    def __init__(self, datasets, transform=None, pairs_per_video=1, frame_range=100, causal=False):
        super().__init__()

        self.datasets = datasets

        self.transform = transform
        if transform is not None and not isinstance(transform, list):
            self.transform = [transform, transform]

        self.pairs_per_video = pairs_per_video
        self.frame_range = frame_range
        self.causal = causal

        self.dataset_indices, self.sequence_indices = self._merge_datasets(datasets, pairs_per_video)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError('list index out of range')

        dataset_id = self.dataset_indices[index]
        sequence_id = self.sequence_indices[index]

        img_files, anno = self.datasets[dataset_id][sequence_id]

        rand_z, rand_x = self._sample_pair(len(img_files))
        img_z = cv2.imread(img_files[rand_z], cv2.IMREAD_COLOR)
        img_x = cv2.imread(img_files[rand_x], cv2.IMREAD_COLOR)

        bbox_z = anno[rand_z, :]
        bbox_x = anno[rand_x, :]
        bbox_xprev = anno[max(rand_x - 1, 0), :]  # Previous frame bounding box, to be used as guide

        # Convert to percentage coordinates.
        bbox_z = to_percentage_coords(bbox_z, img_z.shape)
        bbox_x = to_percentage_coords(bbox_x, img_x.shape)
        bbox_xprev = to_percentage_coords(bbox_xprev, img_x.shape)

        if self.transform is not None:
            img_z, bbox_z, _ = self.transform[0](img_z, bbox_z)
            img_x, bbox_x, bbox_xprev = self.transform[1](img_x, bbox_x, bbox_xprev)

        # Convert to RBG image, and scale values to [0, 1].
        img_z = cv2.cvtColor(img_z, cv2.COLOR_BGR2RGB) / 255.
        img_x = cv2.cvtColor(img_x, cv2.COLOR_BGR2RGB) / 255.

        # Convert to PyTorch Tensors (in particular for images, (w, h, c) is transformed to (c, w, h)).
        img_z = torch.from_numpy(img_z).permute(2, 0, 1).float()
        img_x = torch.from_numpy(img_x).permute(2, 0, 1).float()

        bbox_z = torch.from_numpy(bbox_z).float()
        bbox_x = torch.from_numpy(bbox_x).float()
        bbox_xprev = torch.from_numpy(bbox_xprev).float()

        return img_z, img_x, bbox_z, bbox_x, bbox_xprev

    def __len__(self):
        return len(self.sequence_indices)

    def _sample_pair(self, n):
        if self.causal:
            rand_z = np.random.choice(n - 1)
        else:
            rand_z = np.random.choice(n)

        if self.frame_range == 0:
            return rand_z, rand_z

        possible_x = np.arange(
            max(rand_z - self.frame_range, 1),  # Keep one previous frame (so that we can use it as guide)
            rand_z + self.frame_range + 1)
        possible_x = np.intersect1d(possible_x, np.arange(n))
        if self.causal:
            possible_x = possible_x[possible_x > rand_z]
        else:
            possible_x = possible_x[possible_x != rand_z]

        if possible_x.size > 0:
            rand_x = np.random.choice(possible_x)
        else:
            rand_x = n-1  # To avoid errors when the list of possible x is empty

        return rand_z, rand_x

    @staticmethod
    def _merge_datasets(datasets, pairs_per_video):
        dataset_indices = np.concatenate(
            [np.repeat(i, len(dataset) * pairs_per_video) for i, dataset in enumerate(datasets)]).ravel()
        sequences_indices = np.concatenate(
            [np.tile(np.arange(len(dataset)), pairs_per_video) for dataset in datasets]).ravel()

        return dataset_indices, sequences_indices
