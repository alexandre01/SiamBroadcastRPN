import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt; plt.figure(figsize=(20, 10))
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, ConcatDataset
from tensorboardX import SummaryWriter

from datasets import TrackingNet, ImageNetVID, PairSampler, CocoDetection, COCODistractor, COCONegativePair, COCOPositivePair
from transforms import Transform
from utils import generate_anchors, mask_imgs
from utils.bbox_utils import decode
import utils
from utils import visualize
from loss import MultiBoxLoss
import os
from datetime import datetime
import time
from distutils.dir_util import copy_tree


class Trainer(object):

    def __init__(self, net, cfg):

        self.cfg = cfg

        self.net = net
        self.anchors = generate_anchors(cfg)

        if torch.cuda.is_available():
            self.net.cuda()
            self.anchors = self.anchors.cuda()

        # Dataset transform
        transform = [
            Transform(context_amount=cfg.TRAIN.CROP_CONTEXT_AMOUNT_Z, size=cfg.MODEL.Z_SIZE),
            Transform(context_amount=cfg.TRAIN.CROP_CONTEXT_AMOUNT_X, size=cfg.MODEL.X_SIZE,
                      random_translate=True, random_resize=True, motion_blur=True,
                      random_translate_range=cfg.TRAIN.DATA_AUG_TRANSLATE_RANGE,
                      random_resize_scale_min=cfg.TRAIN.DATA_AUG_RESIZE_SCALE_MIN,
                      random_resize_scale_max=cfg.TRAIN.DATA_AUG_RESIZE_SCALE_MAX
                      )
        ]

        # Training dataset
        trackingnet = TrackingNet(cfg.PATH.TRACKINGNET, subset="train", debug_seq=cfg.TRAIN.DEBUG_SEQ)
        imagenet = ImageNetVID(cfg.PATH.ILSVRC, subset="train")
        sampler = PairSampler([trackingnet, imagenet], cfg=cfg, transform=transform, pairs_per_video=cfg.TRAIN.PAIRS_PER_VIDEO,
                              frame_range=cfg.TRAIN.FRAME_RANGE)
        # Distractor dataset
        coco = CocoDetection(cfg.PATH.COCO, cfg.PATH.COCO_ANN_FILE)
        # coco_distractor = COCODistractor(coco, 4000)
        coco_positive = COCOPositivePair(coco, 4000, cfg=cfg, transform=transform)
        coco_negative = COCONegativePair(coco, 12000, cfg=cfg, transform=transform)

        dataset = ConcatDataset([sampler, coco_positive, coco_negative])
        self.dataloader = DataLoader(dataset, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=4, shuffle=True,
                                     pin_memory=True, drop_last=True)

        # Validation dataset
        val_trackingnet = TrackingNet(cfg.PATH.TRACKINGNET, subset="val")
        val_imagenet = ImageNetVID(cfg.PATH.ILSVRC, subset="val")
        validation_sampler = PairSampler([val_trackingnet, val_imagenet], cfg=cfg, transform=transform,
                                         pairs_per_video=1, frame_range=cfg.TRAIN.FRAME_RANGE)
        val_coco_positive = COCOPositivePair(coco, 100, cfg=cfg, transform=transform)
        val_dataset = ConcatDataset([validation_sampler, val_coco_positive])

        if cfg.TRAIN.DEBUG_SEQ >= 0:  # When debugging on a single sequence, the validation is performed on the same one
            val_dataset = PairSampler([trackingnet], cfg=cfg, transform=transform, pairs_per_video=200)

        self.validation_dataloader = DataLoader(val_dataset, batch_size=min(cfg.TRAIN.BATCH_SIZE, 20), num_workers=4,
                                                shuffle=True, pin_memory=True, drop_last=False)

        # Loss
        self.criterion = MultiBoxLoss(self.anchors, cfg)

        self.optimizer = optim.Adam(self.net.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=cfg.TRAIN.SCHEDULER_STEP_SIZE,
                                                   gamma=cfg.TRAIN.SCHEDULER_GAMMA)

        # Summary Writer
        self.run_id = datetime.now().strftime('%b%d_%H-%M-%S')
        if not cfg.DEBUG:
            self.save_config()
            self.save_code()
            self.writer = SummaryWriter(log_dir=os.path.join(cfg.PATH.DATA_DIR, "runs", self.run_id))

        self.start_epoch = 0

        if cfg.TRAIN.RESUME_CHECKPOINT:
            self.start_epoch = utils.load_checkpoint(cfg.TRAIN.RESUME_CHECKPOINT, self.net, self.optimizer)

        if torch.cuda.is_available():
            self.net = nn.DataParallel(self.net)

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

                # 20% black-and-white data-augmentation
                if torch.rand(1) < 0.2:
                    x = x.mean(dim=1, keepdim=True).expand_as(x)
                    z = z.mean(dim=1, keepdim=True).expand_as(z)

                # Adding masks using ground truth bounding boxes
                z_mask, x_mask = mask_imgs(z, z_bbox), mask_imgs(x, xprev_bbox, use_mask=self.cfg.TRAIN.USE_MASK)

                self.optimizer.zero_grad()

                s = time.time()
                loc_pred, conf_pred = self.net.forward(z, z_mask, x, x_mask)
                forward_time = time.time() - s
                loss_loc, loss_cls, pos_accuracy, neg_accuracy, position_err, size_errors, pos_matches = self.criterion((loc_pred, conf_pred), x_bbox)
                loss = loss_cls + self.cfg.TRAIN.LAMBDA * loss_loc

                if loss > 0:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.)  # Prevent exploding gradients
                    self.optimizer.step()

                iter_idx = epoch * epoch_size + batch_idx
                if not self.cfg.DEBUG:
                    self.log_metrics(iter_idx, loss_loc.item(), loss_cls.item(), loss.item(), pos_accuracy,
                                     neg_accuracy, position_err, size_errors, pos_matches, forward_time,
                                     self.optimizer.param_groups[0]['lr'])

            if not self.cfg.DEBUG:
                """Validation."""

                self.net.eval()

                with torch.no_grad():
                    mean_IOU = self.compute_validation_metrics(epoch_size * (epoch + 1))

                is_best = mean_IOU > self.best_IOU
                self.best_IOU = max(mean_IOU, self.best_IOU)

                """Save Checkpoint."""
                utils.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.net.module.state_dict(),  # Save the 'module' layer from a DataParallel model
                    'optimizer': self.optimizer.state_dict(),
                }, self.cfg.PATH.DATA_DIR, run_id=self.run_id, is_best=is_best)

    def log_metrics(self, iter_idx, loss_l, loss_c, loss, pos_accuracy, neg_accuracy, position_error, size_errors,
                    pos_matches, forward_time, lr):

        self.writer.add_scalar("/train/loss/localization", loss_l, iter_idx)
        self.writer.add_scalar("/train/loss/classification", loss_c, iter_idx)
        self.writer.add_scalar("/train/loss/total", loss, iter_idx)
        self.writer.add_scalar("/train/metrics/pos_accuracy", pos_accuracy, iter_idx)
        self.writer.add_scalar("/train/metrics/neg_accuracy", neg_accuracy, iter_idx)

        self.writer.add_scalar("/train/metrics/position_error", position_error, iter_idx)
        self.writer.add_scalar("/train/metrics/w_error", size_errors[0], iter_idx)
        self.writer.add_scalar("/train/metrics/h_error", size_errors[1], iter_idx)

        self.writer.add_scalar("/train/forward_time", forward_time, iter_idx)
        self.writer.add_scalar("/train/pos_matches", pos_matches, iter_idx)
        self.writer.add_scalar("/train/learning_rate", lr, iter_idx)

    def compute_validation_metrics(self, iter_idx):

        IoUs = []

        for i, (z, x, z_bbox, x_bbox, xprev_bbox) in enumerate(self.validation_dataloader):
            if torch.cuda.is_available():
                z, x, = z.cuda(), x.cuda()
                z_bbox, x_bbox, x_bbox, xprev_bbox = z_bbox.cuda(), x_bbox.cuda(), x_bbox.cuda(), xprev_bbox.cuda()

            # 20% black-and-white data-augmentation
            if torch.rand(1) < 0.2:
                x = x.mean(dim=1, keepdim=True).expand_as(x)
                z = z.mean(dim=1, keepdim=True).expand_as(z)

            z_mask, x_mask = mask_imgs(z, z_bbox), mask_imgs(x, xprev_bbox, use_mask=self.cfg.TRAIN.USE_MASK)
            loc_pred, conf_pred = self.net.forward(z, z_mask, x, x_mask)
            best_ids = conf_pred[:, :, 1].argmax(dim=1)
            best_anchors = self.criterion.point_form_anchors[best_ids]
            indices = best_ids.view(-1, 1, 1).expand(-1, -1, 4)
            pred_bboxs = decode(loc_pred.gather(1, indices).squeeze(1), self.anchors[best_ids],
                                self.cfg.MODEL.ANCHOR_VARIANCES)

            IoUs.append(utils.IoUs(x_bbox, pred_bboxs))

            # Display the first 30 images from the validation set.
            if i < 30:
                visualize.plot_pair((z[0].cpu(), z_bbox[0].cpu()), (x[0].cpu(), pred_bboxs[0].cpu()),
                                    gt_box=x_bbox[0].cpu(), prev_bbox=xprev_bbox[0].cpu(),
                                    anchor=best_anchors[0].cpu(), anchor_id=best_ids[0].cpu())
                self.writer.add_image("Image_{}".format(i), visualize.plot_to_tensor(), iter_idx)
                plt.clf()

        IoUs = torch.cat(IoUs)
        self.writer.add_scalar("/validation/metrics/mean_IoU", IoUs.mean(), iter_idx)
        self.writer.add_scalar("/validation/metrics/median_IoU", IoUs.median(), iter_idx)

        return IoUs.mean()

    def save_code(self):
        archive_dir = os.path.join(self.cfg.PATH.DATA_DIR, "archive", self.run_id)
        copy_tree(".", archive_dir)

    def save_config(self):
        config_file = os.path.join(self.cfg.PATH.DATA_DIR, "configs", self.run_id + ".yaml")
        with open(config_file, "a") as f:
            f.write(self.cfg.dump())
