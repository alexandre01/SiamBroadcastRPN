import torch.nn as nn
import torch.nn.functional as F


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


class Refine2(nn.Module):
    def __init__(self, in_dim1, in_dim2, mid_dim, out_dim):
        super(Refine2, self).__init__()
        self.convFS1 = nn.Conv2d(in_dim1, out_dim, kernel_size=3, padding=1)

        self.convFS2 = nn.Conv2d(in_dim1, mid_dim, kernel_size=3, padding=1)
        self.convFS3 = nn.Conv2d(mid_dim, out_dim, kernel_size=3, padding=1)

        self.convFS4 = nn.Conv2d(in_dim2, out_dim, kernel_size=1)

        self.convMM1 = nn.Conv2d(out_dim, mid_dim, kernel_size=3, padding=1)
        self.convMM2 = nn.Conv2d(mid_dim, out_dim, kernel_size=3, padding=1)

    def forward(self, f, pm):
        s = self.convFS1(f)
        sr = self.convFS2(F.relu(s))
        sr = self.convFS3(F.relu(sr))
        s = s + sr

        pm = F.relu(self.convFS4(pm))

        m = s + F.interpolate(pm, size=f.size(-1), mode='bilinear')

        mr = self.convMM1(F.relu(m))
        mr = self.convMM2(F.relu(mr))
        m = m + mr
        return m
