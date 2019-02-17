import torch.nn as nn


class GC(nn.Module):
    def __init__(self, in_dim, dim, out_dim, kh=7, kw=7):
        super(GC, self).__init__()
        self.conv_l1 = nn.Conv2d(in_dim, dim, kernel_size=(kh, 1),
                                 padding=(int(kh / 2), 0))
        self.conv_l2 = nn.Conv2d(dim, out_dim, kernel_size=(1, kw),
                                 padding=(0, int(kw / 2)))
        self.conv_r1 = nn.Conv2d(in_dim, dim, kernel_size=(1, kw),
                                 padding=(0, int(kw / 2)))
        self.conv_r2 = nn.Conv2d(dim, out_dim, kernel_size=(kh, 1),
                                 padding=(int(kh / 2), 0))

    def forward(self, x):
        x_l = self.conv_l2(self.conv_l1(x))
        x_r = self.conv_r2(self.conv_r1(x))
        x = x_l + x_r
        return x
