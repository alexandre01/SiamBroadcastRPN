from math import sqrt as sqrt
from itertools import product as product
import torch
import os
import shutil


def generate_anchors(cfg):
    mean = []
    for k, f in enumerate(cfg.MODEL.FEATURE_MAPS_DIM):
        for i, j in product(range(f), repeat=2):
            f_k = cfg.MODEL.X_SIZE / cfg.MODEL.FEATURE_MAPS_STRIDES[k]
            # unit center x,y
            cx = (j + 0.5) / f_k
            cy = (i + 0.5) / f_k

            # aspect_ratio: 1
            # rel size: min_size
            s_k = cfg.MODEL.ANCHOR_MIN_SIZES[k] / cfg.MODEL.X_SIZE
            mean += [cx, cy, s_k, s_k]

            # aspect_ratio: 1
            # rel size: sqrt(s_k * s_(k+1))
            if cfg.MODEL.ANCHOR_MAX_SIZES[k] != cfg.MODEL.ANCHOR_MIN_SIZES[k]:
                s_k_prime = sqrt(s_k * (cfg.MODEL.ANCHOR_MAX_SIZES[k] / cfg.MODEL.X_SIZE))
                mean += [cx, cy, s_k_prime, s_k_prime]

            # rest of aspect ratios
            for ar in cfg.MODEL.ANCHOR_ASPECT_RATIOS[k]:
                mean += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
                mean += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]

    # Convert to PyTorch Tensor
    output = torch.Tensor(mean).view(-1, 4)
    output.clamp_(max=1, min=0)

    return output


def mask_img(img, bbox, use_mask=True):
    """
    Adds a mask of the input image according to the provided bounding box.
    img: (Tensor) image to be masked
    bbox: (Tensor) bounding box in pointform format.

    Output: 4-channel image tensor
    """

    img_size = img.shape[-2:]

    if use_mask is False:
        return img.new_ones(img_size).unsqueeze(0)

    mask = img.new_zeros(img_size)

    img_size = img.new_tensor(img_size).float().repeat(2)
    bbox_coords = (bbox * img_size).floor()
    bbox_coords = torch.clamp(torch.min(bbox_coords, img_size - 1), min=0).int()
    mask[bbox_coords[1]:bbox_coords[3] + 1, bbox_coords[0]:bbox_coords[2] + 1] = 1

    return mask.unsqueeze(0)


def mask_imgs(imgs, bboxs, use_mask=True):
    """
    Batch-version of mask_img
    """

    batch_size, _, w, h = imgs.shape

    if use_mask is False:
        return imgs.new_ones(batch_size, w, h).unsqueeze(1)

    masks = imgs.new_zeros(batch_size, w, h)

    img_size = imgs.new_tensor([w, h]).float().repeat(2)
    bbox_coords = (bboxs * img_size).floor()
    bbox_coords = torch.clamp(torch.min(bbox_coords, img_size - 1), min=0).int()
    for i in range(batch_size):
        masks[i, bbox_coords[i, 1]:bbox_coords[i, 3] + 1, bbox_coords[i, 0]:bbox_coords[i, 2] + 1] = 1

    return masks.unsqueeze(1)


def save_checkpoint(state, data_dir, run_id=None, is_best=False):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Based on: https://github.com/cs230-stanford/cs230-code-examples
    """

    checkpoint_dir = os.path.join(data_dir, "checkpoints")

    if run_id is not None:
        checkpoint_dir = os.path.join(checkpoint_dir, run_id)

    filepath = os.path.join(checkpoint_dir, 'last.pth.tar')

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    print("Saving checkpoint to: {}".format(filepath))
    torch.save(state, filepath)

    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint_dir, 'best.pth.tar'))


def load_model(file_name, model):

    if not os.path.exists(file_name):
        raise("File doesn't exist {}".format(file_name))

    print("Loading model: {}".format(file_name))

    device = torch.device("cuda")

    model.load_state_dict(torch.load(file_name))
    model.to(device)


def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))

    print("Loading checkpoint: {}".format(checkpoint))

    device = torch.device("cuda")

    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)

    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])

    return checkpoint['epoch']


def IoU(a, b):
    sa = (a[2] - a[0]) * (a[3] - a[1])
    sb = (b[2] - b[0]) * (b[3] - b[1])
    w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = w * h
    return area / (sa + sb - area)


def IoUs(a, b):
    sa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    sb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    w = (torch.min(a[:, 2], b[:, 2]) - torch.max(a[:, 0], b[:, 0])).clamp(min=0)
    h = (torch.min(a[:, 3], b[:, 3]) - torch.max(a[:, 1], b[:, 1])).clamp(min=0)
    area = w * h
    return area / (sa + sb - area)


def inside(p, bbox):
    """
    p: point (x, y)
    bbox: in x1y1x2y2 format

    Returns mask of indices for which the provided point is included in the bounding box
    """
    return (bbox[:, 0] <= p[0]) & (p[0] <= bbox[:, 2]) & (bbox[:, 1] <= p[1]) & (p[1] <= bbox[:, 3])


def inside_margin(p, bbox):
    """
    p: point (x, y)
    bbox: in cxcywh format

    Returns mask of indices for which the provided point is included in the bounding box
    """
    return ((bbox[:, 0] - p[0]).abs() < 0.15 * bbox[:, 2]) & ((bbox[:, 1] - p[1]).abs() < 0.15 * bbox[:, 3])


def compute_accuracy(ground_truth, prediction, cls):
    """
    Compute the class accuracy
    """

    ground_truth_indices = (ground_truth == cls)

    if ground_truth_indices.sum() == 0:  # no ground-truth element of the given class
        return 1.0

    predicted_classes = torch.sort(prediction[ground_truth_indices], descending=True, dim=1)[1][:, 0]
    return (predicted_classes == cls).float().sum() / ground_truth_indices.float().sum()
