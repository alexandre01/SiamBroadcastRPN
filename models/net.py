import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class BaseNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1_m = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=True)

        resnet = models.resnet50(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1  # 1/4, 256
        self.res3 = resnet.layer2  # 1/8, 512
        self.res4 = resnet.layer3  # 1/16, 1024

        # freeze BNs
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                for p in m.parameters():
                    p.requires_grad = False

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, m):
        """
        Input: frame and mask
        """

        x = (x - self.mean) / self.std
        x = self.conv1(x) + self.conv1_m(m)
        x = self.bn1(x)
        x = self.relu(x)  # 1/2, 64
        x = self.maxpool(x)  # 1/4, 64
        x = self.res2(x)  # 1/4, 64
        r3 = self.res3(x)  # 1/8, 128
        x = self.res4(r3)  # 1/16, 256

        return x, r3


class GC(nn.Module):
    def __init__(self, inplanes, planes, kh=7, kw=7):
        super(GC, self).__init__()
        self.conv_l1 = nn.Conv2d(inplanes, 256, kernel_size=(kh, 1),
                                 padding=(int(kh / 2), 0))
        self.conv_l2 = nn.Conv2d(256, planes, kernel_size=(1, kw),
                                 padding=(0, int(kw / 2)))
        self.conv_r1 = nn.Conv2d(inplanes, 256, kernel_size=(1, kw),
                                 padding=(0, int(kw / 2)))
        self.conv_r2 = nn.Conv2d(256, planes, kernel_size=(kh, 1),
                                 padding=(int(kh / 2), 0))

    def forward(self, x):
        x_l = self.conv_l2(self.conv_l1(x))
        x_r = self.conv_r2(self.conv_r1(x))
        x = x_l + x_r
        return x


class Refine(nn.Module):
    def __init__(self, inplanes, planes, outplanes, scale_factor=2):
        super(Refine, self).__init__()
        self.convFS1 = nn.Conv2d(inplanes, outplanes, kernel_size=3, padding=1)
        self.convFS2 = nn.Conv2d(outplanes, planes, kernel_size=3, padding=1)
        self.convFS3 = nn.Conv2d(planes, outplanes, kernel_size=3, padding=1)
        self.convMM1 = nn.Conv2d(outplanes, planes, kernel_size=3, padding=1)
        self.convMM2 = nn.Conv2d(planes, outplanes, kernel_size=3, padding=1)
        self.scale_factor = scale_factor

    def forward(self, f, pm):
        s = self.convFS1(f)
        sr = self.convFS2(F.relu(s))
        sr = self.convFS3(F.relu(sr))
        s = s + sr

        m = s + F.interpolate(pm, scale_factor=self.scale_factor, mode='bilinear')

        mr = self.convMM1(F.relu(m))
        mr = self.convMM2(F.relu(mr))
        m = m + mr
        return m


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.base = BaseNet()

        self.GC = GC(2048, 512)  # 1/16 -> 1/16
        self.convG1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.convG2 = nn.Conv2d(256, 512, kernel_size=3, padding=1)

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
        x, r3 = self.base(x, x_mask)

        x = torch.cat((x, z), dim=1)

        x = self.GC(x)
        r = self.convG1(F.relu(x))
        r = self.convG2(F.relu(r))
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

        return (loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, 2))

    def temple(self, z, z_mask):
        self.z_embedding, _ = self.base(z.unsqueeze(0), z_mask.unsqueeze(0))

    def infer(self, x, x_mask):
        sources = list()
        loc = list()
        conf = list()
        x, r3 = self.base(x.unsqueeze(0), x_mask.unsqueeze(0))

        x = torch.cat((x, self.z_embedding), dim=1)

        x = self.GC(x)
        r = self.convG1(F.relu(x))
        r = self.convG2(F.relu(r))
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

        return loc.view(-1, 4), conf.view(-1, 2)