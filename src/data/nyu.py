import torch
import json
import cv2
from torchvision.transforms import transforms as T
import numpy as np
import os
import torchvision.transforms.functional as TF
from PIL import Image
import h5py
import random


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

    # A workaround for a pytorch bug
    # https://github.com/pytorch/vision/issues/2194
    class ToNumpy:
        def __call__(self, sample):
            return np.array(sample)


class NYUDataset(BaseDataset):
    def __init__(self, args, mode):
        super(NYUDataset, self).__init__()

        if mode != "train" and mode != "val" and mode != "test":
            raise NotImplementedError

        height, width = (240, 320)
        crop_size = (228, 304)
        # crop_size = (224, 224)

        self.height = height
        self.width = width
        self.crop_size = crop_size
        self.num_sample = args.num_sample
        self.data_dir = args.data_dir
        self.mode = mode
        self.augment = args.augment
        self.is_sparse = args.is_sparse
        self.is_coarse = args.is_coarse

        self.K = torch.Tensor(
            [
                5.1885790117450188e02 / 2.0,
                5.1946961112127485e02 / 2.0,
                3.2558244941119034e02 / 2.0 - 8.0,
                2.5373616633400465e02 / 2.0 - 6.0,
            ]
        )

        with open(os.path.join(args.data_dir, "nyu.json")) as json_file:
            json_data = json.load(json_file)
            self.sample_list = json_data[mode]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        path_file = os.path.join(self.data_dir, self.sample_list[idx]["filename"])

        f = h5py.File(path_file, "r")
        rgb_h5 = f["rgb"][:].transpose(1, 2, 0)
        dep_h5 = f["depth"][:]
        if self.is_coarse:
            coarse_h5 = f["coarse"][:]
        f.close()

        rgb = Image.fromarray(rgb_h5, mode="RGB")
        dep = Image.fromarray(dep_h5.astype("float32"), mode="F")
        if self.is_coarse:
            coarse = Image.fromarray(coarse_h5.astype("float32"), mode="F")

        if self.augment and self.mode == "train":
            _scale = np.random.uniform(1.0, 1.5)
            scale = int(self.height * _scale)
            degree = np.random.uniform(-5.0, 5.0)
            flip = np.random.uniform(0.0, 1.0)

            if flip > 0.5:
                rgb = TF.hflip(rgb)
                dep = TF.hflip(dep)

            rgb = TF.rotate(rgb, angle=degree)
            dep = TF.rotate(dep, angle=degree)

            t_rgb = T.Compose(
                [
                    T.Resize(scale),
                    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    T.CenterCrop(self.crop_size),
                    T.ToTensor(),
                    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )

            t_dep = T.Compose(
                [
                    T.Resize(scale),
                    T.CenterCrop(self.crop_size),
                    self.ToNumpy(),
                    T.ToTensor(),
                ]
            )

            rgb = t_rgb(rgb)
            dep = t_dep(dep)
            dep = dep / _scale

            if self.is_coarse:
                coarse = t_dep(coarse)
                coarse = coarse / _scale

        else:
            t_rgb = T.Compose(
                [
                    T.Resize(self.height),
                    T.CenterCrop(self.crop_size),
                    T.ToTensor(),
                    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )

            t_dep = T.Compose(
                [
                    T.Resize(self.height),
                    T.CenterCrop(self.crop_size),
                    self.ToNumpy(),
                    T.ToTensor(),
                ]
            )

            rgb = t_rgb(rgb)
            dep = t_dep(dep)

            if self.is_coarse:
                coarse = t_dep(coarse)

        if self.mode == "train" and self.is_sparse:
            num_sample = random.randint(5, self.num_sample)
        else:
            num_sample = self.num_sample

        if isinstance(num_sample, str) and "keypoints" in num_sample:
            _mask = self.get_key_point(rgb_h5, num_sample.split("_")[-1])
            dep_sp = self.get_sparse_depth(dep, num_sample, _mask)
        else:
            dep_sp = self.get_sparse_depth(dep, num_sample)

        output = {"rgb": rgb, "dep": dep_sp, "gt": dep}
        if self.is_coarse:
            output = {"rgb": rgb, "dep": dep_sp, "gt": dep, "coarse": coarse}

        return output

    def get_sparse_idx(self, width, height, w, h, num, start_x=None, start_y=None):
        if start_x is None:
            start_x = random.randint(0, width - w)
        if start_y is None:
            start_y = random.randint(0, height - h)
        idx_sample = torch.randperm(w * h)[:num]
        grid = torch.meshgrid(
            [torch.arange(0, h, 1) + start_y, torch.arange(0, w, 1) + start_x]
        )
        x_idx = grid[0].reshape(-1)[idx_sample]
        y_idx = grid[1].reshape(-1)[idx_sample]
        return x_idx, y_idx

    def get_key_point(self, rgb_h5, mode="sift"):
        rgb = Image.fromarray(rgb_h5, mode="RGB")
        t_rgb = T.Compose(
            [
                T.Resize(self.height),
                T.CenterCrop(self.crop_size),
            ]
        )
        rgb = t_rgb(rgb)
        rgb = np.array(rgb)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        if mode == "sift":
            detector = cv2.SIFT.create()
        else:
            detector = cv2.ORB.create()
        keypoints = detector.detect(gray)
        height, width, _ = rgb.shape
        mask = torch.zeros([1, height, width])
        for keypoint in keypoints:
            x = round(keypoint.pt[1])
            y = round(keypoint.pt[0])
            mask[:, x, y] = 1.0
        return mask

    def get_sparse_depth(self, dep, num_sample, _mask=None):
        channel, height, width = dep.shape
        assert channel == 1
        if isinstance(num_sample, str):
            if num_sample == "shift_grid":
                start_x = random.randint(0, width - 100)
                start_y = random.randint(0, height - 100)
                grid = torch.meshgrid(
                    [
                        torch.arange(0, 100, 10) + start_y,
                        torch.arange(0, 100, 10) + start_x,
                    ]
                )
                mask = torch.zeros([channel, height, width])
                mask[:, grid[0], grid[1]] = 1.0
                dep_sp = dep * mask.type_as(dep)
            elif "shift_grid_" in num_sample:
                num = int(num_sample[11:])
                x_idx, y_idx = self.get_sparse_idx(width, height, 100, 100, num)
                mask = torch.zeros([channel, height, width])
                mask[:, x_idx, y_idx] = 1.0
                dep_sp = dep * mask.type_as(dep)
            elif num_sample == "uneven_density":
                mask = torch.zeros([channel, height, width])
                x_idx, y_idx = self.get_sparse_idx(width, height, 25, 25, 100)
                mask[:, x_idx, y_idx] = 1.0
                x_idx, y_idx = self.get_sparse_idx(width, height, 100, 100, 400)
                mask[:, x_idx, y_idx] = 1.0
                dep_sp = dep * mask.type_as(dep)
            elif num_sample == "holes":
                mask = torch.zeros([channel, height, width])
                x_idx, y_idx = self.get_sparse_idx(width, height, width, height, 500)
                mask[:, x_idx, y_idx] = 1.0
                x_idx, y_idx = self.get_sparse_idx(width, height, 200, 200, 40000)
                mask[:, x_idx, y_idx] = 0.0
                dep_sp = dep * mask.type_as(dep)
            elif "keypoints" in num_sample:
                assert _mask is not None
                dep_sp = dep * _mask.type_as(dep)
            elif num_sample == 'short_range':
                range = torch.median(dep[dep>0])
                idx_nnz = torch.nonzero((dep.view(-1) > 0.0001) * (dep.view(-1) < range), as_tuple=False)
                num_idx = len(idx_nnz)
                idx_sample = torch.randperm(num_idx)[:250]
                idx_nnz = idx_nnz[idx_sample[:]]
                mask = torch.zeros((channel * height * width))
                mask[idx_nnz] = 1.0
                mask = mask.view((channel, height, width))
                dep_sp = dep * mask.type_as(dep)
            elif num_sample == 'up_fov':
                mask = torch.zeros([channel, height, width])
                ratio = 0.6
                x_idx, y_idx = self.get_sparse_idx(width, height, int(ratio * width), int(ratio*height), int(500 * ratio * ratio), int((1-ratio) / 2*width), int((1-ratio)/2*height))
                mask[:, x_idx, y_idx] = 1.0
                dep_sp = dep * mask.type_as(dep)
        else:
            idx_nnz = torch.nonzero(dep.view(-1) > 0.0001, as_tuple=False)

            num_idx = len(idx_nnz)
            idx_sample = torch.randperm(num_idx)[:num_sample]

            idx_nnz = idx_nnz[idx_sample[:]]

            mask = torch.zeros((channel * height * width))
            mask[idx_nnz] = 1.0
            mask = mask.view((channel, height, width))

            dep_sp = dep * mask.type_as(dep)

        return dep_sp

    class ToNumpy:
        def __call__(self, sample):
            return np.array(sample)
