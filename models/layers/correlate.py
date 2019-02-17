import torch
from torch.nn import functional as F


def correlate(x, z, padding=0):

    out = []
    for i in range(x.size(0)):
        out.append(F.conv2d(x[i].unsqueeze(0), z[i].unsqueeze(0), padding=padding))

    return torch.cat(out, dim=0)
