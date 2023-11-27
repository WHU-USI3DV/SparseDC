import torch
import torch.nn as nn


# 根据预测的深度和真值深度计算不确定度
def compute_uncertainty(pred, gt, ratio=10):
    # u = torch.exp(torch.abs(pred - gt) / ((pred + gt) / ratio + 1e-8))
    u = 1 - torch.exp(-torch.abs(pred - gt) / (gt / ratio + 1e-8))

    return u

class ConvTokenizer(nn.Module):
    def __init__(self, in_chans=3, embed_dim=64, norm_layer=nn.LayerNorm):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(
                in_chans,
                embed_dim // 2,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
            ),
            nn.Conv2d(
                embed_dim // 2,
                embed_dim,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
            ),
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


def conv_bn_relu(ch_in, ch_out, kernel, stride=1, padding=0, bn=True, relu=True):
    assert (kernel % 2) == 1, "only odd kernel is supported but kernel = {}".format(
        kernel
    )

    layers = []
    layers.append(nn.Conv2d(ch_in, ch_out, kernel, stride, padding, bias=not bn))
    if bn:
        layers.append(nn.BatchNorm2d(ch_out))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))

    layers = nn.Sequential(*layers)

    return layers


def convt_bn_relu(
    ch_in, ch_out, kernel, stride=1, padding=0, output_padding=0, bn=True, relu=True
):
    assert (kernel % 2) == 1, "only odd kernel is supported but kernel = {}".format(
        kernel
    )

    layers = []
    layers.append(
        nn.ConvTranspose2d(
            ch_in, ch_out, kernel, stride, padding, output_padding, bias=not bn
        )
    )
    if bn:
        layers.append(nn.BatchNorm2d(ch_out))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))

    layers = nn.Sequential(*layers)

    return layers


class FillConv(nn.Module):
    def __init__(self, channel):
        super(FillConv, self).__init__()
        self.channel = channel
        self.conv_rgb = conv_bn_relu(3, 48, 3, 1, 1, bn=False)
        self.conv_dep = conv_bn_relu(1, 16, 3, 1, 1, bn=False)

        self._trans = nn.Conv2d(48, 16, 1, 1, 0)
        self.fuse_conv1 = nn.Sequential(
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fuse_conv2 = nn.Sequential(
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.Sigmoid(),
        )
        self.fuse_conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fuse_conv4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
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

    def forward(self, rgb, dep):
        f_rgb = self.conv_rgb(rgb)
        f_dep = self.conv_dep(dep)
        f_dep = torch.cat([self._trans(f_rgb), f_dep], dim=1)
        f_dep = self.fuse_conv1(f_dep) * self.fuse_conv2(f_dep)
        f = torch.cat([f_rgb, f_dep], dim=1)
        f = f + self.fuse_conv3(f) * self.fuse_conv4(f)

        return f
