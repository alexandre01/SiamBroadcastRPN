import torch
import torch.nn.functional as F

from . import Tracker
import utils
from torch import optim
from utils.bbox_utils import format_from_to, to_percentage_coords, to_absolute_coords, decode
from transforms.transforms import *
import matplotlib.pyplot as plt
from loss import MultiBoxLoss


class TrackerDefault(Tracker):
    def __init__(self, net, checkpoint, cfg):
        super().__init__("TrackerDefault")

        self.cfg = cfg

        self.net = net
        if checkpoint is not None:
            utils.load_checkpoint(checkpoint, self.net)

        self.net.eval()

        self.anchors = utils.generate_anchors(cfg)

        if torch.cuda.is_available():
            self.net.cuda()
            self.anchors = self.anchors.cuda()

        self.z_transform = Compose([
            ToAbsoluteCoords(),
            Crop(context_amount=cfg.TRAIN.CROP_CONTEXT_AMOUNT_Z, make_square=False),
            ToPercentCoords(),
            Resize(cfg.MODEL.Z_SIZE),
        ])

        self.x_crop = Crop(context_amount=cfg.TRAIN.CROP_CONTEXT_AMOUNT_X, return_rect=True, make_square=True)
        self.x_resize = Resize(size=cfg.MODEL.X_SIZE)

        self.z_crop = Crop(context_amount=cfg.TRAIN.CROP_CONTEXT_AMOUNT_Z, return_rect=True, make_square=False)
        self.z_resize = Resize(size=cfg.MODEL.Z_SIZE)

        self.criterion = MultiBoxLoss(self.anchors, self.cfg)

    def init(self, img, init_rect):
        self.init_size = init_rect[2:]

        # Convert bounding boxes to x1y1x2y2 format.
        bbox = format_from_to(init_rect, "x1y1wh", "x1y1x2y2")

        # Convert to percentage coordinates.
        self.bbox = to_percentage_coords(bbox, img.shape)

        img_z, bbox_z, _ = self.z_transform(img, self.bbox)
        self.z = self.cfg.MODEL.INPUT_RANGE * torch.from_numpy(img_z).permute(2, 0, 1).float().cuda() / 255.
        bbox_z = torch.from_numpy(bbox_z).float().cuda()

        self.net.temple(self.z, utils.mask_img(self.z, bbox_z))

        self.window = self.build_cosine_window()

    def update(self, _img, iter=0):

        bbox_abs = to_absolute_coords(self.bbox, _img.shape)
        crop_img, bbox, _, crop_rect = self.x_crop(_img, bbox_abs)
        bbox = to_percentage_coords(bbox, crop_img.shape)
        img, bbox, _ = self.x_resize(crop_img, bbox)

        x = self.cfg.MODEL.INPUT_RANGE * torch.from_numpy(img).permute(2, 0, 1).float().cuda() / 255.
        bbox = torch.from_numpy(bbox).float().cuda()

        with torch.no_grad():
            loc_pred, conf_pred = self.net.infer(x, utils.mask_img(x, bbox, use_mask=self.cfg.TRAIN.USE_MASK))

        conf_pred = F.softmax(conf_pred, dim=1)[:, 1].cpu()

        conf_pred = conf_pred.numpy()

        pred_bboxs = decode(loc_pred, self.anchors, self.cfg.MODEL.ANCHOR_VARIANCES).cpu().numpy()

        # Map the bounding box coordinates to the entire image space.
        pred_bboxs = to_absolute_coords(pred_bboxs, crop_img.shape)
        pred_bboxs[:, :2] += crop_rect[:2]
        pred_bboxs[:, 2:] += crop_rect[:2]

        bbox_abs = format_from_to(bbox_abs, "x1y1x2y2", "x1y1wh")
        pred_bboxs = format_from_to(pred_bboxs, "x1y1x2y2", "x1y1wh")

        """
        Engineering.
        """
        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 1.0
            sz2 = (w + pad) * (h + pad)
            return np.sqrt(sz2)

        def sz_wh(wh):
            pad = (wh[0] + wh[1]) * 1.0
            sz2 = (wh[0] + pad) * (wh[1] + pad)
            return np.sqrt(sz2)

        if self.cfg.TRACKING.USE_ENGINEERING:
            # size penalty
            s_c = change(sz(pred_bboxs[:, 2], pred_bboxs[:, 3]) / (sz_wh(bbox_abs[2:])))  # scale penalty
            r_c = change((bbox_abs[2] / bbox_abs[3]) / (pred_bboxs[:, 2] / pred_bboxs[:, 3]))  # ratio penalty

            penalty = np.exp(-(r_c * s_c - 1.) * self.cfg.TRACKING.PENALTY_K)
            score = penalty * conf_pred

            # cosine window
            score = score * (1 - self.cfg.TRACKING.WINDOW_INFLUENCE) + self.window * self.cfg.TRACKING.WINDOW_INFLUENCE
        else:
            score = conf_pred

        best_score_id = np.argmax(score)
        pred_bbox = pred_bboxs[best_score_id]

        if self.cfg.TRACKING.USE_ENGINEERING:
            lr = penalty[best_score_id] * conf_pred[best_score_id] * self.cfg.TRACKING.LR
        else:
            lr = 1.0

        pred_bbox[2:] = bbox_abs[2:] * (1 - lr) + pred_bbox[2:] * lr

        # Prevent too large increase or decrease of the bounding box size
        pred_bbox[2:] = np.clip(pred_bbox[2:], self.init_size / 3, 3 * self.init_size)

        # Snap to image boundaries
        pred_bbox[:2] = np.clip(pred_bbox[:2], 0., _img.shape[:2])

        # Save the predicted bbox in percentage x1y1x2y2 format.
        self.bbox = to_percentage_coords(format_from_to(pred_bbox, "x1y1wh", "x1y1x2y2"), _img.shape)

        return pred_bbox

    def build_cosine_window(self):
        N = len(self.cfg.MODEL.FEATURE_MAPS_DIM)

        nb_anchors = []
        for k, f in enumerate(self.cfg.MODEL.FEATURE_MAPS_DIM):
            num_11_anchors = 2 if self.cfg.MODEL.ANCHOR_MAX_SIZES[k] != self.cfg.MODEL.ANCHOR_MIN_SIZES[k] else 1
            nb_anchors.append(num_11_anchors + 2 * len(self.cfg.MODEL.ANCHOR_ASPECT_RATIOS[k]))

        windows = [np.outer(np.hanning(dim), np.hanning(dim)) for dim in self.cfg.MODEL.FEATURE_MAPS_DIM]
        windows = [np.repeat(windows[i].flatten(), nb_anchors[i]) for i in range(N)]

        return np.concatenate(windows)
