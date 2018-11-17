import torch

from . import Tracker
import utils
from transforms import BaseTransform
from utils.bbox_utils import format_from_to, to_percentage_coords, to_absolute_coords, decode
from transforms.base import *


class TrackerGuided(Tracker):
    def __init__(self, net, checkpoint, cfg):
        super().__init__("ReferenceGuidedRPN")

        self.cfg = cfg

        self.net = net
        utils.load_checkpoint(checkpoint, self.net)
        self.net.eval()

        self.anchors = utils.generate_anchors(cfg)

        if torch.cuda.is_available():
            self.net.cuda()
            self.anchors = self.anchors.cuda()

        self.z_transform = Compose([
            ToAbsoluteCoords(),
            Crop(context_amount=1.0),
            ToPercentCoords(),
            Resize(300),
        ])
        self.x_crop = Crop(context_amount=1.0, return_rect=True)
        self.x_resize = Resize(300)

    def init(self, img, init_rect):
        # Convert bounding boxes to x1y1x2y2 format.
        bbox = format_from_to(init_rect, "x1y1wh", "x1y1x2y2")

        # Convert to percentage coordinates.
        self.bbox = to_percentage_coords(bbox, img.shape)

        img, bbox, _ = self.z_transform(img, self.bbox)

        self.z = torch.from_numpy(img).permute(2, 0, 1).float().cuda() / 255.
        bbox = torch.from_numpy(bbox).float().cuda()

        self.net.temple(self.z, utils.mask_img(self.z, bbox))

    def update(self, _img):

        bbox = to_absolute_coords(self.bbox, _img.shape)
        crop_img, bbox, _, crop_rect = self.x_crop(_img, bbox)
        bbox = to_percentage_coords(bbox, crop_img.shape)
        img, bbox, _ = self.x_resize(crop_img, bbox)

        x = torch.from_numpy(img).permute(2, 0, 1).float().cuda() / 255.
        bbox = torch.from_numpy(bbox).float().cuda()
        crop_rect = torch.from_numpy(crop_rect).float().cuda()

        with torch.no_grad():
            loc_pred, conf_pred = self.net.infer(x, utils.mask_img(x, bbox, use_mask=False))

        best_id = conf_pred[:, 1].argmax()
        pred_bbox = decode(loc_pred[best_id].unsqueeze(0), self.anchors[best_id].unsqueeze(0),
                           self.cfg.MODEL.ANCHOR_VARIANCES)

        bbox = pred_bbox.squeeze(0)
        bbox = to_absolute_coords(bbox, crop_img.shape)
        bbox[:2] += crop_rect[:2]
        bbox[2:] += crop_rect[:2]

        bbox = bbox.cpu().numpy()

        self.bbox = to_percentage_coords(bbox, _img.shape)

        return format_from_to(bbox, "x1y1x2y2", "x1y1wh")
