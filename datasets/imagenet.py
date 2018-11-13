from __future__ import absolute_import, division

import os
import glob
import xml.etree.ElementTree as ET
import numpy as np
import six
import random


class ImageNetVID(object):
    """
    ILSVRC 2015 dataset.
    Bounding boxes are in x1y1x2y2 format.
    """

    def __init__(self, root_dir, subset='train', rand_choice=True):

        super(ImageNetVID, self).__init__()
        self.root_dir = root_dir
        self.rand_choice = rand_choice

        if not self._check_integrity():
            raise Exception('Dataset not found or corrupted. ')

        if subset == 'val':
            self.seq_dirs = sorted(glob.glob(os.path.join(
                self.root_dir, 'Data/VID/val/ILSVRC2015_val_*')))
            self.seq_names = [os.path.basename(s) for s in self.seq_dirs]
            self.anno_dirs = [os.path.join(
                self.root_dir, 'Annotations/VID/val', s) for s in self.seq_names]
        elif subset == 'train':
            self.seq_dirs = sorted(glob.glob(os.path.join(
                self.root_dir, 'Data/VID/train/ILSVRC*/ILSVRC*')))
            self.seq_names = [os.path.basename(s) for s in self.seq_dirs]
            self.anno_dirs = [os.path.join(
                self.root_dir, 'Annotations/VID/train',
                *s.split('/')[-2:]) for s in self.seq_dirs]
        else:
            raise Exception('Unknown subset.')

    def __getitem__(self, index):
        if isinstance(index, six.string_types):
            if not index in self.seq_names:
                raise Exception('Sequence {} not found.'.format(index))
            index = self.seq_names.index(index)
        elif self.rand_choice:
            index = np.random.randint(len(self.seq_names))

        anno_files = sorted(glob.glob(
            os.path.join(self.anno_dirs[index], '*.xml')))
        objects = [ET.ElementTree(file=f).findall('object')
                   for f in anno_files]

        # choose the track id randomly
        track_ids, counts = np.unique([obj.find(
            'trackid').text for group in objects for obj in group], return_counts=True)
        track_id = random.choice(track_ids[counts >= 2])

        frames = []
        anno = []
        for f, group in enumerate(objects):
            for obj in group:
                if not obj.find('trackid').text == track_id:
                    continue
                frames.append(f)
                anno.append([
                    int(obj.find('bndbox/xmin').text),
                    int(obj.find('bndbox/ymin').text),
                    int(obj.find('bndbox/xmax').text),
                    int(obj.find('bndbox/ymax').text)])

        img_files = [os.path.join(self.seq_dirs[index], '%06d.JPEG' % f) for f in frames]
        anno = np.array(anno)

        return img_files, anno

    def __len__(self):
        return len(self.seq_names)

    def _check_integrity(self):
        return os.path.isdir(self.root_dir) and len(os.listdir(self.root_dir)) > 0
