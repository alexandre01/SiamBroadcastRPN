import torch.nn as nn
from torchvision import models
import torch


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

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, m=None, use_mask=True):
        """
        Input: frame and mask
        """

        x = (x - self.mean) / self.std
        x = self.conv1(x)

        if use_mask:
            x += self.conv1_m(m)

        x = self.bn1(x)
        x = self.relu(x)  # 1/2, 64
        x = self.maxpool(x)  # 1/4, 64
        x = self.res2(x)  # 1/4, 64
        r3 = self.res3(x)  # 1/8, 128
        x = self.res4(r3)  # 1/16, 256

        return x, r3
