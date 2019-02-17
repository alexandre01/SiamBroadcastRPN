import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import compute_accuracy
from utils.bbox_utils import point_form, jaccard, encode, decode


class MultiBoxLoss(nn.Module):

    def __init__(self, anchors, cfg):
        super().__init__()

        self.anchors = anchors
        self.point_form_anchors = point_form(anchors)
        self.cfg = cfg

        if cfg.TRAIN.REGRESSION_LOSS == "smooth_l1":
            self.regression_loss = F.smooth_l1_loss
        elif cfg.TRAIN.REGRESSION_LOSS == "l1":
            self.regression_loss = F.l1_loss
        else:
            raise Exception("Unknown regression loss.")

    def forward(self, pred, gt_boxes):

        loc_pred, conf_pred = pred
        batch_size = conf_pred.size(0)

        """
        Labels: overlap => th_high          :  1
                th_low < overlap < th_high  : -1
                overlap <= th_low           :  0
        """
        overlaps = jaccard(gt_boxes, self.point_form_anchors)  # Shape: [batch_size, num_anchors]
        labels = (overlaps >= self.cfg.TRAIN.TH_HIGH).long() - ((self.cfg.TRAIN.TH_LOW < overlaps) &
                                                                (overlaps < self.cfg.TRAIN.TH_HIGH)).long()

        pos = labels == 1  # Shape: [batch_size, num_anchors]
        neg = labels == 0  # Shape: [batch_size, num_anchors]

        N = pos.sum().item()

        """Regression loss."""
        # Repeat the anchors on the batch dimension [batch_size, num_anchors, 4], and select only the positive matches
        matched_anchors = self.anchors.expand(batch_size, -1, -1)[pos]  # Shape: [num_pos, 4]
        # Indices of ground-truth boxes corresponding to positives matches
        i = pos.nonzero()[:, 0]  # Shape: [num_pos]
        # Repeat the ground-truth boxes according to the number of positive matches
        gt_boxes_repeat = gt_boxes[i]  # Shape: [num_pos, 4]
        loc_gt = encode(gt_boxes_repeat, matched_anchors, self.cfg.MODEL.ANCHOR_VARIANCES)
        loss_loc = self.regression_loss(loc_pred[pos], loc_gt, reduction="mean")

        """Classification loss."""
        # Hard negative mining, compute intermediate loss. Shape: [batch_size, num_anchors]
        loss_cls = F.cross_entropy(conf_pred.view(-1, 2), pos.long().view(-1), reduction="none").view(batch_size, -1)
        loss_cls[~neg] = 0  # Filter out non negative boxes
        _, loss_idx = loss_cls.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.sum(dim=1, keepdim=True)
        num_neg = torch.clamp(torch.clamp(self.cfg.TRAIN.NEGPOS_RATIO * num_pos, min=10), max=pos.size(1) - 1)
        neg = idx_rank < num_neg.expand_as(idx_rank)  # Update negatives by picking the ones w. highest confidence loss

        # Classification loss including Positive and Negative examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_pred)
        neg_idx = neg.unsqueeze(2).expand_as(conf_pred)
        conf_picked = conf_pred[(pos_idx + neg_idx).gt(0)].view(-1, 2)
        labels_picked = labels[(pos + neg).gt(0)]
        weight_balance = loc_pred.new_tensor([1/3, 1.])
        loss_cls = F.cross_entropy(conf_picked, labels_picked, weight=weight_balance, reduction="mean")

        # Compute accuracy and pixel error metrics
        pos_accuracy = compute_accuracy(labels_picked, conf_picked, 1)
        neg_accuracy = compute_accuracy(labels_picked, conf_picked, 0)

        img_size = self.cfg.MODEL.X_SIZE
        decoded_loc_pred = decode(loc_pred[pos], matched_anchors, self.cfg.MODEL.ANCHOR_VARIANCES)
        position_error = torch.norm((gt_boxes_repeat[:, :2] - decoded_loc_pred[:, :2]) * img_size, dim=1).mean()
        size_errors = (gt_boxes_repeat[:, 2:] - decoded_loc_pred[:, 2:]).abs().mean(dim=0) * img_size

        return loss_loc, loss_cls, pos_accuracy, neg_accuracy, position_error, size_errors, N
