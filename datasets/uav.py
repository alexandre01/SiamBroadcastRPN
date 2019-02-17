from __future__ import absolute_import, print_function

import os
import glob
import numpy as np
import io
import six
from itertools import chain


class UAV(object):

    def __init__(self, root_dir, sequences=None):
        super(UAV, self).__init__()

        self.root_dir = root_dir

        seq_names = [os.path.basename(seq)[:-4] for seq in glob.glob(os.path.join(self.root_dir, "anno", "UAV123", "*.txt"))]
        self.seq_names = seq_names if sequences is None else [seq for seq in seq_names if seq in sequences]
        self.anno_files = [os.path.join(root_dir, "anno", "UAV123", s + ".txt") for s in self.seq_names]
        self.seq_dirs = [os.path.join(root_dir, "data_seq", "UAV123", seq_name) for seq_name in self.seq_names]

    def __getitem__(self, index):

        img_files = sorted(glob.glob(os.path.join(self.seq_dirs[index], '*.jpg')))

        # to deal with different delimeters
        with open(self.anno_files[index], 'r') as f:
            anno = np.loadtxt(io.StringIO(f.read().replace(',', ' ')))
        assert len(img_files) == len(anno)
        assert anno.shape[1] == 4

        return img_files, anno

    def __len__(self):
        return len(self.seq_names)


if __name__ == "__main__":
    from configs import cfg
    uav = UAV(cfg.PATH.UAV, sequences=None)
    uav[0]
