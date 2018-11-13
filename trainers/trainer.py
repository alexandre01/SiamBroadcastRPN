import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.figure(figsize=(20, 10))

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from datasets import TrackingNet, ImageNetVID, PairSampler
from transforms import BaseTransform, Transform
from utils import generate_anchors, mask_imgs
from utils.bbox_utils import decode
import utils
from utils import visualize
from loss import MultiBoxLoss
import os
from datetime import datetime


class Trainer(object):

    def __init__(self, net, cfg):

        self.cfg = cfg

        self.net = net
        self.anchors = generate_anchors(cfg)

        if torch.cuda.is_available():
            self.net = nn.DataParallel(self.net)
            self.net.cuda()
            self.anchors = self.anchors.cuda()

        # Dataset transform
        transform = [
            Transform(context_amount=1.0),
            Transform(context_amount=1.0)
        ]

        # Training dataset
        trackingnet = TrackingNet(cfg.PATH.TRACKINGNET, subset="train", debug_seq=cfg.TRAIN.DEBUG_SEQ)
        imagenet = ImageNetVID(cfg.PATH.ILSVRC, subset="train")
        sampler = PairSampler([trackingnet, imagenet], transform=transform, pairs_per_video=cfg.TRAIN.PAIRS_PER_VIDEO)
        self.dataloader = DataLoader(sampler, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=4, shuffle=True,
                                     pin_memory=True, drop_last=True)

        # Validation dataset
        val_trackingnet = TrackingNet(cfg.PATH.TRACKINGNET, subset="val")
        val_imagenet = ImageNetVID(cfg.PATH.ILSVRC, subset="val")
        validation_sampler = PairSampler([val_trackingnet, val_imagenet], transform=transform, pairs_per_video=1)

        if cfg.TRAIN.DEBUG_SEQ >= 0:  # When debugging on a single sequence, the validation is performed on the same one
            validation_sampler = PairSampler([trackingnet], transform=transform, pairs_per_video=200)

        self.validation_dataloader = DataLoader(validation_sampler, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=4,
                                                shuffle=False, pin_memory=True, drop_last=False)

        # Loss
        self.criterion = MultiBoxLoss(self.anchors, cfg)

        self.optimizer = optim.Adam(self.net.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=cfg.TRAIN.SCHEDULER_STEP_SIZE,
                                                   gamma=cfg.TRAIN.SCHEDULER_GAMMA)

        # Summary Writer
        self.run_id = datetime.now().strftime('%b%d_%H-%M-%S')
        if not cfg.DEBUG:
            self.writer = SummaryWriter(log_dir=os.path.join(cfg.PATH.DATA_DIR, "runs", self.run_id))

        self.start_epoch = 0

        if cfg.TRAIN.RESUME_CHECKPOINT:
            self.start_epoch = utils.load_checkpoint(cfg.TRAIN.RESUME_CHECKPOINT, self.net, self.optimizer)

        self.best_IOU = 0.

    def train(self):

        print("Training model {} with configuration:".format(type(self.net).__name__))
        print(self.cfg)

        for epoch in range(self.start_epoch, self.cfg.TRAIN.NUM_EPOCHS):

            epoch_size = len(self.dataloader)
            print("Epoch {} / {}, {} iterations".format(epoch + 1, self.cfg.TRAIN.NUM_EPOCHS, epoch_size))

            """Training."""
            self.net.train()

            for batch_idx, batch in enumerate(self.dataloader):

                self.scheduler.step()

                z, x, z_bbox, x_bbox, xprev_bbox = batch

                if torch.cuda.is_available():
                    z, x, = z.cuda(), x.cuda()
                    z_bbox, x_bbox, xprev_bbox = z_bbox.cuda(), x_bbox.cuda(), xprev_bbox.cuda()

                # Adding masks using ground truth bounding boxes
                z_mask, x_mask = mask_imgs(z, z_bbox), mask_imgs(x, xprev_bbox)

                self.optimizer.zero_grad()
                pred = self.net.forward(z, z_mask, x, x_mask)
                loss_loc, loss_cls, pos_accuracy, neg_accuracy, position_err, size_errors = self.criterion(pred, x_bbox)
                loss = loss_cls + self.cfg.TRAIN.LAMBDA * loss_loc
                if loss > 0:
                    loss.backward()
                    self.optimizer.step()

                iter_idx = epoch * epoch_size + batch_idx
                if not self.cfg.DEBUG:
                    self.log_metrics(iter_idx, loss_loc.item(), loss_cls.item(), loss.item(), pos_accuracy,
                                     neg_accuracy, position_err, size_errors, self.optimizer.param_groups[0]['lr'])

            """Validation."""
            if not self.cfg.DEBUG:
                self.net.eval()

                with torch.no_grad():
                    mean_IOU = self.compute_validation_metrics(epoch_size * (epoch + 1))

                is_best = mean_IOU > self.best_IOU
                self.best_IOU = max(mean_IOU, self.best_IOU)

            """Save Checkpoint."""
            if not self.cfg.DEBUG:
                utils.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.net.module.state_dict(),  # Save the 'module' layer from a DataParallel model
                    'optimizer': self.optimizer.state_dict(),
                }, self.cfg.PATH.DATA_DIR, run_id=self.run_id, is_best=is_best)

    def log_metrics(self, iter_idx, loss_l, loss_c, loss, pos_accuracy, neg_accuracy, position_error, size_errors, lr):

        self.writer.add_scalar("/train/loss/localization", loss_l, iter_idx)
        self.writer.add_scalar("/train/loss/classification", loss_c, iter_idx)
        self.writer.add_scalar("/train/loss/total", loss, iter_idx)

        self.writer.add_scalar("/train/metrics/pos_accuracy", pos_accuracy, iter_idx)
        self.writer.add_scalar("/train/metrics/neg_accuracy", neg_accuracy, iter_idx)
        self.writer.add_scalar("/train/metrics/position_error", position_error, iter_idx)
        self.writer.add_scalar("/train/metrics/w_error", size_errors[0], iter_idx)
        self.writer.add_scalar("/train/metrics/h_error", size_errors[1], iter_idx)

        self.writer.add_scalar("/train/learning_rate", lr, iter_idx)

    def compute_validation_metrics(self, iter_idx):

        IoUs = []

        for i, (z, x, z_bbox, x_bbox, xprev_bbox) in enumerate(self.validation_dataloader):
            if torch.cuda.is_available():
                z, x, = z.cuda(), x.cuda()
                z_bbox, x_bbox, xprev_bbox = z_bbox.cuda(), x_bbox.cuda(), xprev_bbox.cuda()

            z_mask, x_mask = mask_imgs(z, z_bbox), mask_imgs(x, xprev_bbox)
            loc_pred, conf_pred = self.net.forward(z, z_mask, x, x_mask)
            best_ids = conf_pred[:, :, 1].argmax(dim=1)
            best_anchors = self.criterion.point_form_anchors[best_ids]
            indices = best_ids.view(-1, 1, 1).expand(-1, -1, 4)
            pred_bboxs = decode(loc_pred.gather(1, indices).squeeze(1), self.anchors[best_ids],
                                self.cfg.MODEL.ANCHOR_VARIANCES)

            IoUs.append(utils.IoUs(best_anchors, pred_bboxs))

            # Display the first 10 images from the validation set.
            if i < 10:
                visualize.plot_pair((z[0].cpu(), z_bbox[0].cpu()), (x[0].cpu(), pred_bboxs[0].cpu()),
                                    gt_box=x_bbox[0].cpu(), anchor=best_anchors[0].cpu(), prev_bbox=xprev_bbox[0].cpu())
                self.writer.add_image("Image {}".format(i), visualize.plot_to_tensor(), iter_idx)
                plt.clf()

        IoUs = torch.cat(IoUs)
        self.writer.add_scalar("/validation/metrics/mean_IoU", IoUs.mean(), iter_idx)
        self.writer.add_scalar("/validation/metrics/median_IoU", IoUs.median(), iter_idx)

        return IoUs.mean()
