import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from scipy import io
import os


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            n = m.weight.size(1)
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()


def load_matconvnet(filename):
    mat = io.loadmat(filename)
    net_dot_mat = mat.get('net')
    params = net_dot_mat['params']
    params = params[0][0]
    params_names = params['name'][0]
    params_names_list = [params_names[p][0] for p in range(params_names.size)]
    params_values = params['value'][0]
    params_values_list = [params_values[p] for p in range(params_values.size)]

    return params_names_list, params_values_list


def load_siamfc_from_matconvnet(filename, model):
    assert isinstance(model.branch, (AlexNetV1, AlexNetV2))
    if isinstance(model.branch, AlexNetV1):
        p_conv = 'conv'
        p_bn = 'bn'
        p_adjust = 'adjust_'
    elif isinstance(model.branch, AlexNetV2):
        p_conv = 'br_conv'
        p_bn = 'br_bn'
        p_adjust = 'fin_adjust_bn'

    params_names_list, params_values_list = load_matconvnet(filename)
    params_values_list = [torch.from_numpy(p) for p in params_values_list]
    for l, p in enumerate(params_values_list):
        param_name = params_names_list[l]
        if 'conv' in param_name and param_name[-1] == 'f':
            p = p.permute(3, 2, 0, 1)
        p = torch.squeeze(p)
        params_values_list[l] = p

    net = (
        model.branch.conv1,
        model.branch.conv2,
        model.branch.conv3,
        model.branch.conv4,
        model.branch.conv5)

    for l, layer in enumerate(net):
        layer[0].weight.data[:] = params_values_list[
            params_names_list.index('%s%df' % (p_conv, l + 1))]
        layer[0].bias.data[:] = params_values_list[
            params_names_list.index('%s%db' % (p_conv, l + 1))]

        if l < len(net) - 1:
            layer[1].weight.data[:] = params_values_list[
                params_names_list.index('%s%dm' % (p_bn, l + 1))]
            layer[1].bias.data[:] = params_values_list[
                params_names_list.index('%s%db' % (p_bn, l + 1))]

            bn_moments = params_values_list[
                params_names_list.index('%s%dx' % (p_bn, l + 1))]
            layer[1].running_mean[:] = bn_moments[:, 0]
            layer[1].running_var[:] = bn_moments[:, 1] ** 2
        elif model.norm.norm == 'bn':
            model.norm.bn.weight.data[:] = params_values_list[
                params_names_list.index('%sm' % p_adjust)]
            model.norm.bn.bias.data[:] = params_values_list[
                params_names_list.index('%sb' % p_adjust)]

            bn_moments = params_values_list[
                params_names_list.index('%sx' % p_adjust)]
            model.norm.bn.running_mean[:] = bn_moments[0]
            model.norm.bn.running_var[:] = bn_moments[1] ** 2
        elif model.norm.norm == 'linear':
            model.norm.linear.weight.data[:] = params_values_list[
                params_names_list.index('%sf' % p_adjust)]
            model.norm.linear.bias.data[:] = params_values_list[
                params_names_list.index('%sb' % p_adjust)]

    return model


class XCorr(nn.Module):

    def __init__(self):
        super(XCorr, self).__init__()

    def forward(self, z, x):
        out = []
        for i in range(z.size(0)):
            out.append(F.conv2d(x[i, :].unsqueeze(0),
                                z[i, :].unsqueeze(0)))

        return torch.cat(out, dim=0)


class Adjust2d(nn.Module):

    def __init__(self, norm='bn'):
        super(Adjust2d, self).__init__()
        assert norm in [None, 'bn', 'cosine', 'euclidean', 'linear']
        self.norm = norm
        if norm == 'bn':
            self.bn = nn.BatchNorm2d(1)
        elif norm == 'linear':
            self.linear = nn.Conv2d(1, 1, 1, bias=True)
        self._initialize_weights()

    def forward(self, out, z=None, x=None):
        if self.norm == 'bn':
            out = self.bn(out)
        elif self.norm == 'linear':
            out = self.linear(out)
        elif self.norm == 'cosine':
            n, k = out.size(0), z.size(-1)
            norm_z = torch.sqrt(
                torch.pow(z, 2).view(n, -1).sum(1)).view(n, 1, 1, 1)
            norm_x = torch.sqrt(
                k * k * F.avg_pool2d(torch.pow(x, 2), k, 1).sum(1, keepdim=True))
            out = out / (norm_z * norm_x + 1e-32)
            out = (out + 1) / 2
        elif self.norm == 'euclidean':
            n, k = out.size(0), z.size(-1)
            sqr_z = torch.pow(z, 2).view(n, -1).sum(1).view(n, 1, 1, 1)
            sqr_x = k * k * \
                F.avg_pool2d(torch.pow(x, 2), k, 1).sum(1, keepdim=True)
            out = out + sqr_z + sqr_x
            out = out.clamp(min=1e-32).sqrt()
        elif self.norm == None:
            out = out

        return out

    def _initialize_weights(self):
        if self.norm == 'bn':
            self.bn.weight.data.fill_(1)
            self.bn.bias.data.zero_()
        elif self.norm == 'linear':
            self.linear.weight.data.fill_(1e-3)
            self.linear.bias.data.zero_()


class AlexNetV1(nn.Module):

    def __init__(self):
        super(AlexNetV1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, groups=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, groups=2),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, groups=2))
        initialize_weights(self)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        return x


class AlexNetV2(nn.Module):

    def __init__(self):
        super(AlexNetV2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, groups=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 1))
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, groups=2),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 32, 3, 1, groups=2))
        initialize_weights(self)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        return x


class SiamFC(nn.Module):

    def __init__(self, root_pretrained):
        super(SiamFC, self).__init__()
        self.root_pretrained = root_pretrained

        self.branch = AlexNetV2()
        self.norm = Adjust2d(norm="bn")
        self.xcorr = XCorr()

        self.load_weights()

    def load_weights(self):
        net_path = os.path.join(self.root_pretrained, "baseline-conv5_e55.mat")
        load_siamfc_from_matconvnet(net_path, self)

    def forward(self, z, x):
        assert z.size()[:2] == x.size()[:2]
        z = self.branch(z)
        x = self.branch(x)
        out = self.xcorr(z, x)
        out = self.norm(out, z, x)

        return out
