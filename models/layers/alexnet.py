import torch.nn as nn
from torchvision import models
import torch


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.featureExtract = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=11, stride=2),
            nn.BatchNorm2d(192),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 512, kernel_size=5),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 768, kernel_size=3),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            nn.Conv2d(768, 768, kernel_size=3),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            nn.Conv2d(768, 512, kernel_size=3),
            nn.BatchNorm2d(512),
        )

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, m=None, use_mask=True):
        """
        Input: frame and mask
        """

        x = (x - self.mean) / self.std
        x = self.featureExtract(x)

        return x
