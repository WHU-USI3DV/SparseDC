# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.utils import compute_uncertainty, conv_bn_relu

class GateConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GateConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self._conv1 = nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        self._conv2 = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self._conv3 = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 3, 1, 1),
            nn.Sigmoid(),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        for m in self.modules():
            # 判断是否属于Conv2d
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                # 判断是否有偏置
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.3)
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0.1)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, f):
        f = self._conv1(f)

        return f + self._conv2(f) * self._conv3(f)


def adapt_pool(m, dep):
    dep = F.avg_pool2d(dep, 3, 2, 1)
    m = F.avg_pool2d(m.float(), 3, 2, 1)
    dep = (m > 0) * dep / (m + 1e-8)

    return dep


def get_depth_pool(dep):
    masks, depths = [], []
    m = dep > 0
    masks.append(m)
    depths.append(dep)
    for _ in range(3):
        dep = adapt_pool(m, dep)
        m = dep > 0
        masks.append(m)
        depths.append(dep)

    return masks, depths


class PlaceHolder(nn.Module):
    def __init__(self):
        super(PlaceHolder, self).__init__()

    def forward(self, x):
        return x


class ECA(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class FuseConv(nn.Module):
    def __init__(
        self, in_features, out_features, is_gate_fuse=False, ratio=1, stride=4
    ):
        super(FuseConv, self).__init__()

        self.is_gate_fuse = is_gate_fuse
        self._tran1 = nn.Conv2d(in_features, in_features, 1, 1, 0)
        self._tran2 = nn.Conv2d(in_features, in_features, 1, 1, 0)

        self._fuse = GateConv(in_features * 2, in_features)

        self._net = conv_bn_relu(in_features, out_features, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        for m in self.modules():
            # 判断是否属于Conv2d
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                # 判断是否有偏置
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.3)
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0.1)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, f1, f2, s1, s2):
        s1 = torch.clamp(s1, 0.01, 0.99)  # 保留部分
        s2 = torch.clamp(s2, 0.01, 0.99)

        f1 = self._tran1(f1)
        f2 = self._tran2(f2)

        if self.is_gate_fuse:
            x = self._fuse(torch.cat([(1 - s1) * f1, (1 - s2) * f2], dim=1))
        else:
            x = self._fuse(torch.cat([f1, f2], dim=1))

        return self._net(x)


class UDConv(nn.Module):
    def __init__(self, in_features, mlp_dim=64, out_channel=64, is_first=False):
        super(UDConv, self).__init__()

        self.is_first = is_first

        if self.is_first:
            self._net = nn.Sequential(
                conv_bn_relu(in_features, in_features // 2, 3, 1, 1),
                conv_bn_relu(in_features // 2, in_features // 2, 3, 1, 1),
                conv_bn_relu(in_features // 2, out_channel, 3, 1, 1),
            )
            self.u_conv = nn.Sequential(
                nn.Conv2d(out_channel, mlp_dim, 1, 1, 0),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(mlp_dim, 1, 1, 1, 0),
                nn.Sigmoid(),
            )
            self.d_conv = nn.Sequential(
                nn.Conv2d(out_channel, mlp_dim, 1, 1, 0),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(mlp_dim, 1, 1, 1, 0),
                nn.ReLU(inplace=True),
            )
        else:
            self._net = nn.Sequential(
                conv_bn_relu(in_features + 2, in_features // 2, 3, 1, 1),
                conv_bn_relu(in_features // 2, in_features // 2, 3, 1, 1),
                conv_bn_relu(in_features // 2, out_channel, 3, 1, 1),
            )
            self.u_conv = nn.Sequential(
                nn.Conv2d(out_channel, mlp_dim, 1, 1, 0),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(mlp_dim, 1, 1, 1, 0),
            )
            self.d_conv = nn.Sequential(
                nn.Conv2d(out_channel, mlp_dim, 1, 1, 0),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(mlp_dim, 1, 1, 1, 0),
            )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        for m in self.modules():
            # 判断是否属于Conv2d
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                # 判断是否有偏置
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.3)
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0.1)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, u=None, d=None):
        if self.is_first:
            x = self._net(x)
            d = self.d_conv(x)
            u = self.u_conv(x)
        else:
            x = self._net(torch.cat([x, d, u], dim=1))
            d = d + self.d_conv(x)
            u = u + self.u_conv(x)

        d = torch.clamp(d, 0.0)
        u = torch.clamp(u, 0.0, 1.0)

        return x, u, d


class UncertaintyFuse_(nn.Module):
    def __init__(
        self,
        l_in_channels,
        g_in_channels,
        bot_channel,
        max_depth=10.0,
        mlp_channel=64,
        out_channel=64,
        fuse_channel=128,
        last_replace=True,
        is_gate_fuse=False,
        **kwargs
    ):
        super(UncertaintyFuse_, self).__init__()
        self.max_depth = max_depth
        self.mlp_channel = mlp_channel
        self.l_in_channels = l_in_channels
        self.g_in_channels = g_in_channels
        self.bot_channel = bot_channel
        self.last_replace = last_replace
        self.is_gate_fuse = is_gate_fuse
        self.num_layers = 6  # [1/1, 1/2, 1/4, 1/8, 1/16, 1/32]

        # 轻量化
        self.fuse_channels = [self.g_in_channels[0]] + self.l_in_channels
        self.fuse_channels = [min(fuse_channel, c) for c in self.fuse_channels]

        self.fuse_convs = nn.ModuleList()
        self.local_skip_convs = nn.ModuleList()
        self.glocal_skip_convs = nn.ModuleList()
        self.local_bot_convs = nn.ModuleList()
        self.glocal_bot_convs = nn.ModuleList()
        self.fuse_bot_convs = nn.ModuleList()

        self.ud_convs = nn.ModuleList()
        self.g_ud_convs = nn.ModuleList()
        self.l_ud_convs = nn.ModuleList()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        for i in range(6):
            f_c = self.fuse_channels[i]
            if i > 0 and i < 4:
                g_bot_conv = nn.Conv2d(f_c + self.fuse_channels[i - 1], f_c, 3, 1, 1)
            else:
                g_bot_conv = nn.Conv2d(f_c, f_c, 3, 1, 1)
            if i > 1:
                if i > 2:
                    g_ud_conv = UDConv(f_c, mlp_channel, out_channel, False)
                else:
                    g_ud_conv = UDConv(f_c, mlp_channel, out_channel, True)
            else:
                g_ud_conv = PlaceHolder()
            if i < 4:
                g_c = self.g_in_channels[i]
                gloabl_skip_conv = nn.Conv2d(g_c, f_c, 1, 1, 0)
            else:
                gloabl_skip_conv = nn.Conv2d(self.fuse_channels[i - 1], f_c, 1, 1, 0)

            if i > 0:
                l_c = self.l_in_channels[i - 1]
                local_skip_conv = nn.Conv2d(l_c, f_c, 1, 1, 0)
                if i > 1:
                    fuse_conv = FuseConv(f_c, f_c, self.is_gate_fuse)
                    if i > 2:
                        l_ud_conv = UDConv(f_c, mlp_channel, out_channel, False)
                        ud_conv = UDConv(f_c, mlp_channel, out_channel, False)
                        fuse_bot_conv = nn.Conv2d(
                            f_c + self.fuse_channels[i - 1], f_c, 3, 1, 1
                        )
                    else:
                        l_ud_conv = UDConv(f_c, mlp_channel, out_channel, True)
                        ud_conv = UDConv(f_c, mlp_channel, out_channel, True)
                        fuse_bot_conv = nn.Conv2d(f_c, fuse_channel, 3, 1, 1)
                    l_bot_conv = nn.Conv2d(
                        f_c + self.fuse_channels[i - 1], f_c, 3, 1, 1
                    )
                else:
                    l_bot_conv = nn.Conv2d(f_c, f_c, 3, 1, 1)
                    ud_conv = PlaceHolder()
                    l_ud_conv = PlaceHolder()
                    fuse_conv = PlaceHolder()
            else:
                local_skip_conv = PlaceHolder()
                l_ud_conv = PlaceHolder()
                l_bot_conv = PlaceHolder()
                fuse_bot_conv = PlaceHolder()
                fuse_conv = PlaceHolder()
                ud_conv = PlaceHolder()

            self.glocal_skip_convs.append(gloabl_skip_conv)
            self.glocal_bot_convs.append(g_bot_conv)
            self.g_ud_convs.append(g_ud_conv)
            self.local_skip_convs.append(local_skip_conv)
            self.l_ud_convs.append(l_ud_conv)
            self.local_bot_convs.append(l_bot_conv)
            self.fuse_convs.append(fuse_conv)
            self.ud_convs.append(ud_conv)
            self.fuse_bot_convs.append(fuse_bot_conv)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        for m in self.modules():
            # 判断是否属于Conv2d
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                # 判断是否有偏置
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.3)
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0.1)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, outs, depth):
        local_f, global_f = outs

        l_prev, g_prev, f_prev, u, d = None, None, None, None, None
        l_u, l_d, g_u, g_d = None, None, None, None
        depths = {"local_d": [], "global_d": [], "fuse_d": []}
        uncertainties = {"local_u": [], "global_u": [], "fuse_u": []}
        masks, sparse_d = get_depth_pool(depth)

        # 转过来
        global_f = global_f[::-1]
        local_f = local_f[::-1]
        masks = masks[::-1]
        sparse_d = sparse_d[::-1]

        for i in range(self.num_layers):
            if i > 0:
                if i > 1:
                    _d = sparse_d[i - 2]
                    _m = masks[i - 2]
                    if i > 3:
                        g_prev = self.glocal_bot_convs[i](
                            self.glocal_skip_convs[i](g_prev)
                        )
                    else:
                        g_prev = self.glocal_bot_convs[i](
                            torch.cat(
                                [self.glocal_skip_convs[i](global_f[i]), g_prev], dim=1
                            )
                        )
                    l_prev = self.local_bot_convs[i](
                        torch.cat(
                            [self.local_skip_convs[i](local_f[i - 1]), l_prev], dim=1
                        )
                    )

                    # get u, d, f
                    _, g_u, g_d = self.g_ud_convs[i](g_prev, g_u, g_d)
                    _, l_u, l_d = self.l_ud_convs[i](l_prev, l_u, l_d)
                    _lu = compute_uncertainty(l_d, _d)
                    _lu = l_u * ~_m + _lu * _m
                    _gu = compute_uncertainty(g_d, _d)
                    _gu = g_u * ~_m + _gu * _m

                    f = self.fuse_convs[i](l_prev, g_prev, _lu, _gu)

                    if i > 2:
                        f_prev = self.fuse_bot_convs[i](torch.cat([f, f_prev], dim=1))
                    else:
                        f_prev = self.fuse_bot_convs[i](f)

                    if i > 4:
                        f_prev, u, d = self.ud_convs[i](f_prev, u, d)
                        if self.last_replace:
                            u = u * ~_m + 0.05 * _m
                            d = d * ~_m + _d * _m
                    else:
                        _, u, d = self.ud_convs[i](f_prev, u, d)

                    depths["fuse_d"].append(d)
                    depths["local_d"].append(l_d)
                    depths["global_d"].append(g_d)
                    uncertainties["fuse_u"].append(u)
                    uncertainties["local_u"].append(l_u)
                    uncertainties["global_u"].append(g_u)

                    if i < 5:
                        u = u * ~_m + 0.05 * _m
                        d = d * ~_m + _d * _m
                        f_prev = self.up(f_prev)
                        l_prev = self.up(l_prev)
                        g_prev = self.up(g_prev)
                        u = self.up(u)
                        l_u = self.up(_lu)
                        g_u = self.up(_gu)
                        d = self.up(d)
                        l_d = self.up(l_d)
                        g_d = self.up(g_d)
                else:
                    g_prev = self.glocal_bot_convs[i](
                        torch.cat(
                            [self.glocal_skip_convs[i](global_f[i]), g_prev], dim=1
                        )
                    )
                    l_prev = self.local_bot_convs[i](
                        self.local_skip_convs[i](local_f[i - 1])
                    )
                    g_prev = self.up(g_prev)
                    l_prev = self.up(l_prev)
            else:
                g_prev = self.glocal_bot_convs[i](
                    self.glocal_skip_convs[i](global_f[i])
                )
                g_prev = self.up(g_prev)  # 1 / 32 - > 1 / 16

        return f_prev, depths, uncertainties

