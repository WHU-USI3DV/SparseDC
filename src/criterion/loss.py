from importlib import import_module
import torch
import torch.nn as nn
import numpy as np


def get(args):
    loss_name = args.model_name + 'Loss'
    module_name = 'loss.' + loss_name.lower()
    module = import_module(module_name)

    return getattr(module, loss_name)


class BaseLoss:

    def __init__(self, args):
        self.args = args

        self.loss_dict = {}
        self.loss_module = nn.ModuleList()

        # Loss configuration : w1*l1+w2*l2+w3*l3+...
        # Ex : 1.0*L1+0.5*L2+...
        for loss_item in args.loss.split('+'):
            weight, loss_type = loss_item.split('*')

            module_name = 'src.criterion.loss'
            module = import_module(module_name)
            loss_func = getattr(module, loss_type + 'Loss')(args)

            loss_tmp = {'weight': float(weight), 'func': loss_func}

            self.loss_dict.update({loss_type: loss_tmp})
            self.loss_module.append(loss_func)

        self.loss_dict.update({'Total': {'weight': 1.0, 'func': None}})

    def __call__(self, sample, output):
        return self.compute(sample, output)

    def cuda(self, gpu):
        self.loss_module.cuda(gpu)

    def compute(self, sample, output):
        loss_val = []
        for idx, loss_type in enumerate(self.loss_dict):
            loss = self.loss_dict[loss_type]
            loss_func = loss['func']
            if loss_func is not None:
                loss_tmp = loss['weight'] * loss_func(sample, output)
                loss_val.append(loss_tmp)

        loss_val = torch.cat(loss_val, dim=1)
        loss_sum = torch.sum(loss_val)

        return loss_sum, loss_val


class DepthLoss(BaseLoss):

    def __init__(self, args):
        super(DepthLoss, self).__init__(args)

        self.loss_name = []

        for k, _ in self.loss_dict.items():
            self.loss_name.append(k)

    def compute(self, pred, gt):
        loss_val = {}
        loss_sum = torch.tensor(0.0)

        for _, loss_type in enumerate(self.loss_dict):
            loss = self.loss_dict[loss_type]
            loss_func = loss['func']
            if loss_func is None:
                continue

            loss_tmp = loss_func(pred, gt)

            loss_tmp = loss['weight'] * loss_tmp
            loss_sum = loss_sum + loss_tmp
            loss_val[f"{loss_type}_loss"] = loss_tmp
        loss_val['total_loss'] = loss_sum

        return loss_sum, loss_val


class L1Loss(nn.Module):

    def __init__(self, args):
        super(L1Loss, self).__init__()

        self.args = args
        self.t_valid = 0.0001

    def forward(self, pred, gt):
        gt = torch.clamp(gt, min=0, max=self.args.max_depth)
        pred = torch.clamp(pred, min=0, max=self.args.max_depth)

        valid_mask = (gt > self.t_valid).detach()
        diff = gt - pred
        diff = diff[valid_mask]
        loss = diff.abs().mean()
        return loss


class L2Loss(nn.Module):

    def __init__(self, args):
        super(L2Loss, self).__init__()

        self.args = args
        self.t_valid = 0.0001

    def forward(self, pred, gt):
        gt = torch.clamp(gt, min=0, max=self.args.max_depth)
        pred = torch.clamp(pred, min=0, max=self.args.max_depth)

        valid_mask = (gt > self.t_valid).detach()
        diff = gt - pred
        diff = diff[valid_mask]
        loss = (diff**2).mean()

        return loss


class SiLogLoss(nn.Module):

    def __init__(self, args, lambd=0.5):
        super().__init__()
        self.args = args
        self.lambd = lambd

    def forward(self, pred, target):
        valid_mask = (target > 0).detach()
        diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask])
        loss = torch.sqrt(
            torch.pow(diff_log, 2).mean() -
            self.lambd * torch.pow(diff_log.mean(), 2))

        return loss


class EdgeLoss(nn.Module):

    def __init__(self, args):
        super(EdgeLoss, self).__init__()
        self.args = args

    def forward(self, pred, gt):
        '''
        Computes the local smoothness loss
        Arg(s):
            predict : torch.Tensor[float32]
                N x 1 x H x W predictions
            image : torch.Tensor[float32]
                N x 1 x H x W RGB image
        Returns:
            torch.Tensor[float32] : mean SSIM distance between source and target images
        '''
        gt = torch.clamp(gt, min=0, max=self.args.max_depth)

        predict_dy, predict_dx = gradient_yx(pred)
        image_dy, image_dx = gradient_yx(gt)

        # Create edge awareness weights
        weights_x = torch.exp(
            -torch.mean(torch.abs(image_dx), dim=1, keepdim=True))
        weights_y = torch.exp(
            -torch.mean(torch.abs(image_dy), dim=1, keepdim=True))

        smoothness_x = torch.mean(weights_x * torch.abs(predict_dx))
        smoothness_y = torch.mean(weights_y * torch.abs(predict_dy))

        return smoothness_x + smoothness_y


'''
Helper functions for constructing loss functions
'''


def gradient_yx(T):
    '''
    Computes gradients in the y and x directions
    Arg(s):
        T : torch.Tensor[float32]
            N x C x H x W tensor
    Returns:
        torch.Tensor[float32] : gradients in y direction
        torch.Tensor[float32] : gradients in x direction
    '''

    dx = T[:, :, :, :-1] - T[:, :, :, 1:]
    dy = T[:, :, :-1, :] - T[:, :, 1:, :]
    return dy, dx


class Sobel(nn.Module):

    def __init__(self):
        super(Sobel, self).__init__()
        self.edge_conv = nn.Conv2d(1,
                                   2,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   bias=False)
        edge_kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        edge_ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        edge_k = np.stack((edge_kx, edge_ky))

        edge_k = torch.from_numpy(edge_k).float().view(2, 1, 3, 3)
        self.edge_conv.weight = nn.Parameter(edge_k)

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = self.edge_conv(x)
        out = out.contiguous().view(-1, 2, x.size(2), x.size(3))

        return out


class GradLoss(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.get_gradient = Sobel()

    def forward(self, pred, target):
        target = torch.clamp(target, min=0, max=self.args.max_depth)

        valid_mask = (target > 0).detach()
        depth_grad = self.get_gradient(pred)
        output_grad = self.get_gradient(target)
        depth_grad_dx = depth_grad[:, 0, :, :].contiguous().view_as(pred)
        depth_grad_dy = depth_grad[:, 1, :, :].contiguous().view_as(pred)
        output_grad_dx = output_grad[:, 0, :, :].contiguous().view_as(pred)
        output_grad_dy = output_grad[:, 1, :, :].contiguous().view_as(pred)
        loss_dx = torch.log(
            torch.abs((output_grad_dx - depth_grad_dx)[valid_mask]) +
            1.0).mean()
        loss_dy = torch.log(
            torch.abs((output_grad_dy - depth_grad_dy)[valid_mask]) +
            1.0).mean()
        loss = loss_dx + loss_dy

        return loss
