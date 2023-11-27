import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.utils import compute_uncertainty, FillConv
from timm.models.layers import to_2tuple


class DepthPooling(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.num_layers = num_layers

    def forward(self, d):
        output = []
        for i in range(self.num_layers):
            _, _, H, W = d.size()
            if (H % 2 != 0) or (W % 2 != 0):
                d = F.pad(d, (0, W % 2, 0, H % 2))
            if i == 0:
                d = F.avg_pool2d(d, 4, 4, 0)
            else:
                d = F.avg_pool2d(d, 2, 2, 0)
            output.append(d)

        return output


class Uncertainty_(nn.Module):
    def __init__(
        self,
        backbone_l=None,
        backbone_g=None,
        decode=None,
        refiner=None,
        criterion=None,
        is_padding=True,
        padding_size=320,
        max_depth=10.0,
        channels=64,
        pretrain=None,
        ratio=10.0,
        is_fill=False,
        **kwargs,
    ):
        super(Uncertainty_, self).__init__()

        self.backbone_l = backbone_l
        self.backbone_g = backbone_g
        self.decode = decode
        self.refiner = refiner
        self.criterion = criterion
        self.num_features = self.backbone_l.num_features
        self.num_layers = len(self.num_features)
        self.is_padding = is_padding
        self.padding_size = to_2tuple(padding_size)
        self.max_depth = max_depth
        self.pretrain = pretrain
        self.ratio = ratio
        self.is_fill = is_fill

        if self.pretrain is not None:
            self.load_pretrain(self.pretrain)

        if self.is_fill:
            self.fill_conv = FillConv(64)
            self.d_conv = nn.Sequential(
                nn.Conv2d(64, 64, 1, 1, 0),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 1, 1, 1, 0),
                nn.ReLU(inplace=True),
            )

        if self.refiner is not None:
            self.guide_layer = nn.Sequential(
                nn.Conv2d(channels, channels, 3, 1, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(channels, self.refiner.ch_g, 3, 1, 1),
            )

    def forward(self, sample):
        if not self.training:
            gt = sample['dep']
        else:
            gt = sample['gt']

        if self.is_padding:
            rgb = self.padding(sample["rgb"])
            dep = self.padding(sample["dep"])
        else:
            rgb = sample["rgb"]
            dep = sample["dep"]

        dep = torch.clamp(dep, min=0.0, max=self.max_depth)
        gt = torch.clamp(gt, min=0.0, max=self.max_depth)

        mode = "train" if self.training else "val"

        if self.is_fill:
            f = self.fill_conv(rgb, dep)
            f_depth = self.d_conv(f)
            outs = (self.backbone_l(rgb, dep, f), self.backbone_g(rgb, dep, f))
        else:
            outs = (self.backbone_l(rgb, dep), self.backbone_g(rgb, dep))

        x, depths, uncertainties = self.decode(outs, dep)

        if self.is_padding:
            dep = self.depadding(dep)
            rgb = self.depadding(rgb)
            if self.is_fill:
                f_depth = self.depadding(f_depth)

        loss, loss_val = self.get_loss(gt, depths, uncertainties, dep)
        if self.is_fill:
            f_depth = torch.clamp(f_depth, min=0.0, max=self.max_depth)
            m = (gt > 0).detach()
            f_d_loss = ((f_depth - gt) ** 2)[m].mean()
            loss = loss + f_d_loss * 0.05
            loss_val[f"{mode}/ori_dloss"] = f_d_loss

        if self.is_padding:
            pred_init = self.depadding(depths["fuse_d"][-1])
            conf = 1 - self.depadding(uncertainties["fuse_u"][-1])
        else:
            pred_init = depths["fuse_d"][-1]
            conf = 1 - uncertainties["fuse_u"][-1]
        if self.refiner is not None:
            guide = self.guide_layer(x)
            if self.refiner.__class__.__name__ == "NLSPN":
                if self.is_padding:
                    guide = self.depadding(guide)
                depth = self.refiner(pred_init, guide, conf, dep, rgb)
            depth = torch.clamp(depth, min=0.0, max=self.max_depth)

            refine_loss, refine_loss_val = self.criterion(depth, gt)
            loss = loss + refine_loss
            for key in refine_loss_val.keys():
                loss_val[f"{mode}/refine_{key}"] = refine_loss_val[key]
            return depth, loss, loss_val
        else:
            pred_init = torch.clamp(pred_init, min=0.0, max=self.max_depth)
            return pred_init, loss, loss_val

    def padding(self, x):
        _, _, H, W = x.shape
        self.H_pad = (self.padding_size[0] - H) // 2
        self.W_pad = (self.padding_size[1] - W) // 2
        x = torch.nn.functional.pad(
            x, pad=[self.W_pad, self.W_pad, self.H_pad, self.H_pad]
        )

        return x

    def depadding(self, x):
        _, _, H, W = x.shape
        x = x[
            ..., self.H_pad : H - self.H_pad, self.W_pad : W - self.W_pad
        ].contiguous()

        return x

    def get_loss(self, dep, depths, uncertainties, i_dep):
        def compute_loss(prefix):
            d = depths[f"{prefix}_d"][idx]
            u = uncertainties[f"{prefix}_u"][idx]
            u_ = compute_uncertainty(d, dep, self.ratio)
            u_loss = torch.abs(u - u_)[m].mean()
            d_loss = ((d - dep) ** 2)[m].mean()
            _loss = d_loss + eps * u_loss
            loss_val[f"{mode}/level_{i}_{prefix}_dloss"] = d_loss
            loss_val[f"{mode}/level_{i}_{prefix}_uloss"] = u_loss
            return _loss

        if self.is_padding:
            dep = self.padding(dep)
            i_dep = self.padding(i_dep)
        num_layers = len(depths["fuse_d"])
        loss = torch.tensor(0.0)
        afa = 0.8
        mode = "train" if self.training else "val"
        loss_val = {}
        eps = 0.5
        for i in range(num_layers):
            idx = num_layers - i - 1
            m = (dep > 0.0).detach()
            i_m = (i_dep > 0).detach()
            l_loss = compute_loss("local")
            g_loss = compute_loss("global")
            f_loss = compute_loss("fuse")
            loss = loss + afa**i * (0.5 * (l_loss + g_loss) + f_loss)
            dep = self.adapt_pool(m, dep)
            i_dep = self.adapt_pool(i_m, i_dep)

        return loss, loss_val

    def load_pretrain(self, ckpt_path):
        ckpt = torch.load(ckpt_path, "cpu")["state_dict"]
        pretrained_dict = {}
        for k, v in self.state_dict().items():
            if f"net.{k}" not in ckpt:
                pretrained_dict[k] = v
            else:
                pretrained_dict[k] = ckpt[f"net.{k}"]
        self.load_state_dict(pretrained_dict)
        print("load pretrain is done !")
        del ckpt, pretrained_dict
        torch.cuda.empty_cache()

    def adapt_pool(self, m, dep):
        dep = F.avg_pool2d(dep, 3, 2, 1)
        m = F.avg_pool2d(m.float(), 3, 2, 1)
        dep = (m > 0) * dep / (m + 1e-8)

        return dep

