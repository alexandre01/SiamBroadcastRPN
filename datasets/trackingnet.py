import numpy as np
import os
from os import scandir


class TrackingNet(object):
    """
    TrackingNet dataset.
    Bounding boxes are in x1y1wh format.
    """

    def __init__(self, root_dir, subset="train", debug_seq=-1):
        self.root_dir = root_dir

        validation_chunk = "TRAIN_5"
        validation_size = 300

        val_chunk_seq_names = [validation_chunk + ":" + f.name for f in
                               scandir(os.path.join(self.root_dir, validation_chunk, "frames")) if f.is_dir()]

        if subset == "val":
            self.seq_names = val_chunk_seq_names[:validation_size]
        elif subset == "train":
            chunks = [f.name for f in scandir(self.root_dir) if f.is_dir() and "TRAIN" in f.name
                      and f.name != validation_chunk]
            self.seq_names = [chunk + ":" + f.name for chunk in chunks for f in
                              scandir(os.path.join(self.root_dir, chunk, "frames")) if f.is_dir() ]
            self.seq_names.extend(val_chunk_seq_names[validation_size:])
        else:
            raise Exception('Unknown subset.')

        if debug_seq >= 0:
            self.seq_names = [self.seq_names[debug_seq]]

    def __len__(self):
        return len(self.seq_names)

    def __getitem__(self, idx):

        chunk, seq_id = self.seq_names[idx].split(":")

        num_image_files = len([f.name for f in scandir(os.path.join(self.root_dir, chunk, "frames", seq_id))])
        image_files = [os.path.join(self.root_dir, chunk, "frames", seq_id, "{}.jpg".format(i)) for i in range(num_image_files)]
        anno = np.loadtxt(os.path.join(self.root_dir, chunk, "anno", seq_id + ".txt"), delimiter=",", )

        # Convert bounding boxes to x1y1x2y2 format.
        anno[:, 2] += anno[:, 0]
        anno[:, 3] += anno[:, 1]

        return image_files, anno
