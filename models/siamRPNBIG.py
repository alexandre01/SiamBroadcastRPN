import torch
import torch.nn as nn
import torch.nn.functional as F


"""
Original SiamRPN network.
During training: use forward(z, x).
For tracking: call temple() once on the exemplar, and use infer on the search images afterwards.
"""


def correlate(x, z):
    out = []
    for i in range(x.size(0)):
        out.append(F.conv2d(x[i].unsqueeze(0), z[i]))

    return torch.cat(out, dim=0)


class SiamRPNBIG(nn.Module):
    def __init__(self, cfg, feat_in=512, feature_out=512, anchor=5):
        super(SiamRPNBIG, self).__init__()
        self.anchor = anchor
        self.feature_out = feature_out
        self.featureExtract = nn.Sequential(
            nn.Conv2d(3, 192, 11, stride=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(192, 512, 5),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(512, 768, 3),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),

            nn.Conv2d(768, 768, 3),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),

            nn.Conv2d(768, 512, 3),
            nn.BatchNorm2d(512),
        )

        self.cfg = cfg

        # Regression branch
        self.conv_r1 = nn.Conv2d(feat_in, feature_out * 4 * anchor, 3)
        self.conv_r2 = nn.Conv2d(feat_in, feature_out, 3)

        # Classification branch
        self.conv_cls1 = nn.Conv2d(feat_in, feature_out * 2 * anchor, 3)
        self.conv_cls2 = nn.Conv2d(feat_in, feature_out, 3)

        self.regress_adjust = nn.Conv2d(4 * anchor, 4 * anchor, 1)

        self.r1_kernel = []
        self.cls1_kernel = []

        # Load pretrained AlexNet weights
        self.reset_params()
        self.freeze_params()

    def reset_params(self):
        model_dict = self.state_dict()
        model_dict.update(torch.load(self.cfg.PATH.ALEXNETBIG_WEIGHTS))
        self.load_state_dict(model_dict)

    def load_pretrained(self):
        self.load_state_dict(torch.load(self.cfg.PATH.PRETRAINED_MODEL))

    def freeze_params(self):
        # As stated in the paper, freeze the first 3 conv layers.
        for i in [0, 4, 8]:
            for p in self.featureExtract[i].parameters():
                p.requires_grad = False

        # Set the associated batch norm layers to evaluation mode.
        for i in [1, 5, 9]:
            self.featureExtract[i].requires_grad = False
            self.featureExtract[i].eval()

    def infer(self, x):
        x_f = self.featureExtract(x)
        return self.regress_adjust(F.conv2d(self.conv_r2(x_f), self.r1_kernel)), \
               F.conv2d(self.conv_cls2(x_f), self.cls1_kernel)

    def temple(self, z):
        z_f = self.featureExtract(z)

        r1_kernel_raw = self.conv_r1(z_f)
        cls1_kernel_raw = self.conv_cls1(z_f)
        kernel_size = r1_kernel_raw.data.size()[-1]

        self.r1_kernel = r1_kernel_raw.view(self.anchor * 4, self.feature_out, kernel_size, kernel_size)
        self.cls1_kernel = cls1_kernel_raw.view(self.anchor * 2, self.feature_out, kernel_size, kernel_size)

    def forward(self, z, x):
        z_f = self.featureExtract(z)
        x_f = self.featureExtract(x)

        r1_kernel_raw = self.conv_r1(z_f)
        cls1_kernel_raw = self.conv_cls1(z_f)

        batch_size, kernel_size = z.size(0), r1_kernel_raw.size(-1)

        r1_kernel = r1_kernel_raw.view(batch_size, self.anchor * 4, self.feature_out, kernel_size, kernel_size)
        cls1_kernel = cls1_kernel_raw.view(batch_size, self.anchor * 2, self.feature_out, kernel_size, kernel_size)

        return (self.regress_adjust(correlate(self.conv_r2(x_f), r1_kernel)),
                correlate(self.conv_cls2(x_f), cls1_kernel))
