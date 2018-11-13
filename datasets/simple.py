from __future__ import absolute_import

import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from utils.bbox_utils import to_percentage_coords


class SimpleSampler(Dataset):

    def __init__(self, base_dataset, transform=None, pairs_per_video=1):
        super().__init__()

        self.base_dataset = base_dataset
        self.transform = transform
        self.pairs_per_video = pairs_per_video

        self.indices = np.arange(len(self.base_dataset), dtype=int)
        self.indices = np.tile(self.indices, pairs_per_video)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError('list index out of range')

        index = self.indices[index]
        img_files, anno = self.base_dataset[index]

        rand = np.random.choice(len(img_files))
        img = cv2.imread(img_files[rand], cv2.IMREAD_COLOR)
        bbox = anno[rand, :]

        # Convert to percentage coordinates.
        bbox = to_percentage_coords(bbox, img.shape)

        if self.transform is not None:
            img, bbox = self.transform(img, bbox)

        # Convert to RBG image, and scale values to [0, 1].
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.

        # Convert to PyTorch Tensors (in particular for images, (w, h, c) is transformed to (c, w, h)).
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        bbox = torch.from_numpy(bbox).float()
        return img, bbox

    def __len__(self):
        return len(self.indices)
