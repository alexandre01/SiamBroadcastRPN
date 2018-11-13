import torch

from . import Tracker
import utils
from transforms import BaseTransform
from utils.bbox_utils import format_from_to, to_percentage_coords, to_absolute_coords, decode


class TrackerGuided(Tracker):
    def __init__(self, net, checkpoint, cfg):
        super().__init__("Guided")

        self.cfg = cfg

        self.net = net
        utils.load_checkpoint(checkpoint, self.net)
        self.net.eval()

        self.anchors = utils.generate_anchors(cfg)

        if torch.cuda.is_available():
            self.net.cuda()
            self.anchors = self.anchors.cuda()

        # TODO: replace this temporary base transform
        self.transform = BaseTransform()

    def init(self, img, init_rect):
        # Convert bounding boxes to x1y1x2y2 format.
        bbox = format_from_to(init_rect, "x1y1wh", "x1y1x2y2")

        # Convert to percentage coordinates.
        bbox = to_percentage_coords(bbox, img.shape)

        img, _, _ = self.transform(img, None)

        self.z = torch.from_numpy(img).permute(2, 0, 1).float().cuda()
        self.bbox = torch.from_numpy(bbox).float().cuda()

        self.net.temple(self.z, utils.mask_img(self.z, self.bbox))

    def update(self, img):

        img, _, _ = self.transform(img, None)

        x = torch.from_numpy(img).permute(2, 0, 1).float().cuda()

        with torch.no_grad():
            loc_pred, conf_pred = self.net.infer(x, utils.mask_img(x, self.bbox))

        best_id = conf_pred[:, 1].argmax()
        pred_bbox = decode(loc_pred[best_id].unsqueeze(0), self.anchors[best_id].unsqueeze(0),
                           self.cfg.MODEL.ANCHOR_VARIANCES)

        self.bbox = pred_bbox.squeeze(0)

        bbox = to_absolute_coords(pred_bbox, x.shape).cpu().numpy()
        return format_from_to(bbox, "x1y1x2y2", "x1y1wh")
