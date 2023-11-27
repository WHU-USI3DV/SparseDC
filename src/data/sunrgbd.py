import torch

from torchvision.transforms import transforms as T
import numpy as np
import os
import torchvision.transforms.functional as TF
from PIL import Image


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


class SUNDataset(BaseDataset):
    def __init__(self, args, mode):
        super(SUNDataset, self).__init__()

        if mode != "train" and mode != "val" and mode != "test":
            raise NotImplementedError

        height, width = (240, 320)
        crop_size = (228, 304)

        self.height = height
        self.width = width
        self.crop_size = crop_size

        self.num_sample = args.num_sample
        self.data_dir = args.data_dir
        self.mode = mode
        self.radio = args.radio
        self.augment = args.augment

        if self.mode == "train" or self.mode == "val":
            self.file_name = os.listdir(os.path.join(self.data_dir, "train_depth"))
        else:
            self.file_name = os.listdir(os.path.join(self.data_dir, "test_depth"))

        self.file_name.sort()

        if self.mode == "train" or self.mode == "val":
            num = len(self.file_name) - int(len(self.file_name) * self.radio)
            import random

            random.seed(0)
            random.shuffle(self.file_name)
            if self.mode == "train":
                self.file_name = self.file_name[:num]
            else:
                self.file_name = self.file_name[num:]

    def __len__(self):
        return len(self.file_name)

    def __getitem__(self, idx):
        rgb_filename = "img-{:06d}.jpg".format(int(self.file_name[idx][:-4]))
        if self.mode == "train" or self.mode == "val":
            rgb_path = os.path.join(self.data_dir, "train_images", rgb_filename)
            depth_path = os.path.join(self.data_dir, "train_depth", self.file_name[idx])
        else:
            rgb_path = os.path.join(self.data_dir, "test_images", rgb_filename)
            depth_path = os.path.join(
                self.data_dir, "test_depth_gt", self.file_name[idx]
            )
            input_path = os.path.join(
                self.data_dir, "test_depth_input", self.file_name[idx]
            )
        rgb = Image.open(rgb_path).convert("RGB")
        dep = Image.open(depth_path)
        dep_sp = Image.open(input_path)

        if self.augment and self.mode == "train":
            _scale = np.random.uniform(1.0, 1.5)
            scale = np.int(self.height * _scale)
            degree = np.random.uniform(-5.0, 5.0)
            flip = np.random.uniform(0.0, 1.0)

            if flip > 0.5:
                rgb = TF.hflip(rgb)
                dep = TF.hflip(dep)

            rgb = TF.rotate(rgb, angle=degree, resample=Image.NEAREST)
            dep = TF.rotate(dep, angle=degree, resample=Image.NEAREST)

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
            dep = t_dep(dep) / 10000.0
            dep_sp = t_dep(dep_sp) / 10000.0

        output = {"rgb": rgb, "dep": dep_sp, "gt": dep}

        return output

    def get_sparse_depth(self, dep, num_sample):
        channel, height, width = dep.shape

        assert channel == 1

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


if __name__ == "__main__":
    data_loader = SUNDataset("a", mode="train")
    max_depth = []
    for batch in data_loader:
        print(batch["gt"].max())
        max_depth.append(batch["gt"].max())
    print("max:", max(max_depth))
