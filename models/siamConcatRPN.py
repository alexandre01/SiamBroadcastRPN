import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import *
from utils import utils, bbox_utils


class SiamConcatRPN(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.base = BaseNet()

        self.cfg = cfg
        self.use_mask = cfg.TRAIN.USE_MASK
        self.use_correlation_guide = cfg.TRACKING.USE_CORRELATION_GUIDE

        self.GC1 = GC(2048, 512, 512, kh=11, kw=11)
        self.convG1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.GC2 = GC(512, 256, 512, kh=9, kw=9)
        self.convG2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.GC3 = GC(512, 256, 512, kh=7, kw=7)
        self.convG3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.RF = Refine(512, 256, 512)

        self.extras = nn.ModuleList([
            nn.Conv2d(512, 256, kernel_size=1),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(512, 128, kernel_size=1),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(256, 128, kernel_size=1),
            nn.Conv2d(128, 256, kernel_size=3),
            nn.Conv2d(256, 128, kernel_size=1),
            nn.Conv2d(128, 256, kernel_size=3)
        ])

        self.loc = nn.ModuleList([
            nn.Conv2d(512, 16, kernel_size=3, padding=1),
            nn.Conv2d(512, 24, kernel_size=3, padding=1),
            nn.Conv2d(512, 24, kernel_size=3, padding=1),
            nn.Conv2d(256, 24, kernel_size=3, padding=1),
            nn.Conv2d(256, 16, kernel_size=3, padding=1),
            nn.Conv2d(256, 16, kernel_size=3, padding=1)
        ])

        self.conf = nn.ModuleList([
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            nn.Conv2d(512, 12, kernel_size=3, padding=1),
            nn.Conv2d(512, 12, kernel_size=3, padding=1),
            nn.Conv2d(256, 12, kernel_size=3, padding=1),
            nn.Conv2d(256, 8, kernel_size=3, padding=1),
            nn.Conv2d(256, 8, kernel_size=3, padding=1)
        ])

    def forward(self, z, z_mask, x, x_mask):
        sources = list()
        loc = list()
        conf = list()

        z, _ = self.base(z, z_mask)
        x, r3 = self.base(x, x_mask, use_mask=self.use_mask)

        x = torch.cat((x, z), dim=1)

        x = self.GC1(x)
        r = self.convG1(F.relu(x))
        x = x + r
        x = self.GC2(F.relu(x))
        r = self.convG2(F.relu(x))
        x = x + r
        x = self.GC3(F.relu(x))
        r = self.convG3(F.relu(x))
        x = x + r

        x = self.RF(r3, x)

        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        return loc.view(loc.size(0), -1, 4), conf.view(conf.size(0), -1, 2)

    def temple(self, z, z_mask):

        if self.use_correlation_guide:
            self.anchors = utils.generate_anchors(self.cfg).cuda()
            self.point_form_anchors = bbox_utils.point_form(self.anchors).cuda()
            self.siamFC = SiamFC(root_pretrained=self.cfg.PATH.PRETRAINED_SIAMFC).cuda()
            self.siamFC.train()

            self.z_cropped = z.unsqueeze(0)[:, :, 75:75 + 151, 75:75 + 151]

        self.z_embedding, _ = self.base(z.unsqueeze(0), z_mask.unsqueeze(0))

    def infer(self, x, x_mask):
        sources = list()
        loc = list()
        conf = list()

        if self.use_correlation_guide:
            correlation = self.siamFC(self.z_cropped * 255., x.unsqueeze(0) * 255.)

            padding = 11
            map_dim = 38
            full_map_dim = 60
            index = correlation.argmax()
            i, j = (padding + index // map_dim).float() / full_map_dim, (padding + index % map_dim).float() / full_map_dim
            anchor_indices = utils.inside((j, i), self.point_form_anchors)

        x, r3 = self.base(x.unsqueeze(0), x_mask.unsqueeze(0), use_mask=self.use_mask)

        x = torch.cat((x, self.z_embedding), dim=1)

        x = self.GC1(x)
        r = self.convG1(F.relu(x))
        x = x + r
        x = self.GC2(F.relu(x))
        r = self.convG2(F.relu(x))
        x = x + r
        x = self.GC3(F.relu(x))
        r = self.convG3(F.relu(x))
        x = x + r

        x = self.RF(r3, x)

        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.use_correlation_guide:
            conf = conf.view(-1, 2)
            conf[~anchor_indices, 0] = 1e5
            conf[~anchor_indices, 1] = -1e5

        return loc.view(-1, 4), conf.view(-1, 2)
