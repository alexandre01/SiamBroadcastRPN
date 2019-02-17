import cv2
import numpy as np
import time
from utils.openvot_viz import show_frame


class Tracker(object):
    def __init__(self, name, image_mode="RGB"):
        self.name = name
        self.image_mode = image_mode

    def init(self, image, init_rect):
        raise NotImplementedError()

    def update(self, image, iter):
        raise NotImplementedError()

    def track(self, img_files, init_rect, visualize=False):
        frame_num = len(img_files)
        bndboxes = np.zeros((frame_num, 4))
        bndboxes[0, :] = init_rect
        speed_fps = np.zeros(frame_num)

        for f, img_file in enumerate(img_files):

            image = cv2.imread(img_file, cv2.IMREAD_COLOR)

            if self.image_mode == "RGB":
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            start_time = time.time()
            if f == 0:
                self.init(image, init_rect)
            else:
                bndboxes[f, :] = self.update(image, f)
            elapsed_time = time.time() - start_time
            speed_fps[f] = 1. / elapsed_time

            if visualize:
                show_frame(image, bndboxes[f, :], fig_n=1)

        return bndboxes, speed_fps

from .tracker import TrackerDefault
from .siamRPNBIG import TrackerSiamRPNBIG


def load_tracker(net, checkpoint, cfg):
    if checkpoint == "":
        checkpoint = None

    if cfg.MODEL.NET == "SiamRPNBIG":
        return TrackerSiamRPNBIG(net, checkpoint, cfg)
    else:
        return TrackerDefault(net, checkpoint, cfg)
