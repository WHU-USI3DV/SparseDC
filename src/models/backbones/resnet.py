"""
    GraphCSPN: Geometry-Aware Depth Completion via Dynamic GCNs

    European Conference on Computer Vision (ECCV) 2022

    The code is based on https://github.com/zzangjinsun/NLSPN_ECCV20
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision
from src.models.utils import conv_bn_relu


class ResNet(nn.Module):

    def __init__(self, model_name='resnet34', **kwargs):
        super(ResNet, self).__init__()

        # Encoder
        self.conv_rgb = conv_bn_relu(3, 64, 3, 2, 1, bn=False)

        self.model_name = model_name

        if self.model_name == 'resnet18':
            net = torchvision.models.resnet18(pretrained=True)
            self.num_features = [64, 128, 256, 512, 512]
        elif self.model_name == 'resnet34':
            net = torchvision.models.resnet34(pretrained=True)
            self.num_features = [64, 128, 256, 512, 512]
        elif self.model_name == 'resnet50':
            net = torchvision.models.resnet50(pretrained=True)
            self.num_features = [256, 512, 1024, 2048, 2048]
        elif self.model_name == 'resnext50':
            net = torchvision.models.resnext50_32x4d(
                torchvision.models.ResNeXt50_32X4D_Weights.DEFAULT)
            self.num_features = [256, 512, 1024, 2048, 2048]
        else:
            return RuntimeError

        # 1/1
        self.conv1 = net.layer1
        # 1/2
        self.conv2 = net.layer2
        # 1/4
        self.conv3 = net.layer3
        # 1/8
        self.conv4 = net.layer4

        del net

        # 1/16
        self.conv5 = conv_bn_relu(self.num_features[-1],
                                  self.num_features[-1],
                                  kernel=3,
                                  stride=2,
                                  padding=1)

        self.convs = nn.ModuleList(
            [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5])

    def forward(self, rgb):

        # f0 = self.conv_rgb(rgb).permute(0, 3, 1, 2).contiguous()
        f = self.conv_rgb(rgb)
        outs = []
        outs.append(f)

        for conv in self.convs:
            f = conv(f)
            outs.append(f)

        # f1 = self.conv1(f0)
        # f2 = self.conv2(f1)
        # f3 = self.conv3(f2)
        # f4 = self.conv4(f3)
        # f5 = self.conv5(f4)
        # return [f1, f2, f3, f4, f5]

        return outs


class ResNetU_(nn.Module):

    def __init__(self, model_name='resnet34', is_fill=False, **kwargs):
        super(ResNetU_, self).__init__()

        # Encoder
        self.is_fill = is_fill
        if self.is_fill:
            self.conv = conv_bn_relu(64, 64, 3, 1, 1)
        else:
            self.conv_rgb = conv_bn_relu(3, 48, 3, 1, 1, bn=False)
            self.conv_d = conv_bn_relu(1, 16, 3, 1, 1, bn=False)
            self.conv = conv_bn_relu(64, 64, 3, 1, 1, bn=False)

        self.model_name = model_name

        if self.model_name == 'resnet18':
            net = torchvision.models.resnet18(
                torchvision.models.ResNet18_Weights.DEFAULT)
            self.num_features = [64, 128, 256, 512, 512]
        elif self.model_name == 'resnet34':
            net = torchvision.models.resnet34(
                torchvision.models.ResNet34_Weights.DEFAULT)
            self.num_features = [64, 128, 256, 512, 512]
        elif self.model_name == 'resnet50':
            net = torchvision.models.resnet50(
                torchvision.models.ResNet50_Weights.DEFAULT)
            self.num_features = [256, 512, 1024, 2048, 2048]
        elif self.model_name == 'resnext50':
            net = torchvision.models.resnext50_32x4d(
                torchvision.models.ResNeXt50_32X4D_Weights.DEFAULT)
            self.num_features = [256, 512, 1024, 2048, 2048]
        else:
            return RuntimeError

        # 1/1
        conv1 = net.layer1
        # 1/2
        conv2 = net.layer2
        # 1/4
        conv3 = net.layer3
        # 1/8
        conv4 = net.layer4

        del net

        # 1 / 16
        conv5 = conv_bn_relu(self.num_features[-1], self.num_features[-1], 3,
                             2, 1)

        self.convs = nn.ModuleList([conv1, conv2, conv3, conv4, conv5])

    def forward(self, rgb, dep, f=None):
        if not self.is_fill:
            f = torch.cat([self.conv_rgb(rgb), self.conv_d(dep)], dim=1)
        f = self.conv(f)
        outs = []
        for conv in self.convs:
            f = conv(f)
            outs.append(f)

        return outs
