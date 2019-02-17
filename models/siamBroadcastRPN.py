import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import *


class SiamBroadcastRPN(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.resnetX = ResNet()
        self.resnetZ = ResNet()

        self.cfg = cfg

        self.relation = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.extras = nn.ModuleList([
            nn.Conv2d(256, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

        ])

        self.loc = nn.ModuleList([
            nn.Conv2d(256, 24, kernel_size=3, padding=1),
            nn.Conv2d(256, 24, kernel_size=3, padding=1)
        ])

        self.conf = nn.ModuleList([
            nn.Conv2d(256, 12, kernel_size=3, padding=1),
            nn.Conv2d(256, 12, kernel_size=3, padding=1),
        ])

    def forward(self, z, z_mask, x, x_mask):
        sources = list()
        loc = list()
        conf = list()

        z = self.resnetZ(z)
        x = self.resnetX(x)

        z = F.max_pool2d(z, kernel_size=8)
        z = z.expand_as(x)

        x = torch.cat((x, z), dim=1)

        x = self.relation(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = v(x)
            if k % 5 == 4:
                sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        return loc.view(loc.size(0), -1, 4), conf.view(conf.size(0), -1, 2)

    def temple(self, z, z_mask):
        z = self.resnetZ(z.unsqueeze(0))
        z = F.max_pool2d(z, kernel_size=8)
        self.z = z.expand(-1, -1, 32, 32)

    def infer(self, x, x_mask):
        sources = list()
        loc = list()
        conf = list()

        x = self.resnetX(x.unsqueeze(0))

        x = torch.cat((x, self.z), dim=1)

        x = self.relation(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = v(x)
            if k % 5 == 4:
                sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        return loc.view(-1, 4), conf.view(-1, 2)
