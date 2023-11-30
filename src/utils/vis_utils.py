import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

if not ("DISPLAY" in os.environ):
    import matplotlib as mpl

    mpl.use("Agg")
import cv2

cmap = plt.cm.jet
cmap2 = plt.cm.nipy_spectral


def validcrop(img):
    ratio = 256 / 1216
    h = img.size()[2]
    w = img.size()[3]
    return img[:, :, h - int(ratio * w) :, :]


def depth_colorize(depth):
    depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    depth = 255 * cmap(depth)[:, :, :3]  # H, W, C
    return depth.astype("uint8")


def feature_colorize(feature):
    feature = (feature - np.min(feature)) / ((np.max(feature) - np.min(feature)))
    feature = 255 * cmap2(feature)[:, :, :3]
    return feature.astype("uint8")


def mask_vis(mask):
    mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))
    mask = 255 * mask
    return mask.astype("uint8")


def kitti_merge_into_row(
    ele, pred, predrgb=None, predg=None, extra=None, extra2=None, extrargb=None
):
    def preprocess_depth(x):
        y = np.squeeze(x.data.cpu().numpy())
        return depth_colorize(y)

    # if is gray, transforms to rgb
    img_list = []
    if "rgb" in ele:
        rgb = np.squeeze(ele["rgb"][0, ...].data.cpu().numpy())
        rgb = np.transpose(rgb, (1, 2, 0))
        img_list.append(rgb)
    elif "g" in ele:
        g = np.squeeze(ele["g"][0, ...].data.cpu().numpy())
        g = np.array(Image.fromarray(g).convert("RGB"))
        img_list.append(g)
    if "d" in ele:
        img_list.append(preprocess_depth(ele["d"][0, ...]))
        img_list.append(preprocess_depth(pred[0, ...]))
    if extrargb is not None:
        img_list.append(preprocess_depth(extrargb[0, ...]))
    if predrgb is not None:
        predrgb = np.squeeze(ele["rgb"][0, ...].data.cpu().numpy())
        predrgb = np.transpose(predrgb, (1, 2, 0))
        # predrgb = predrgb.astype('uint8')
        img_list.append(predrgb)
    if predg is not None:
        predg = np.squeeze(predg[0, ...].data.cpu().numpy())
        predg = mask_vis(predg)
        predg = np.array(Image.fromarray(predg).convert("RGB"))
        # predg = predg.astype('uint8')
        img_list.append(predg)
    if extra is not None:
        extra = np.squeeze(extra[0, ...].data.cpu().numpy())
        extra = mask_vis(extra)
        extra = np.array(Image.fromarray(extra).convert("RGB"))
        img_list.append(extra)
    if extra2 is not None:
        extra2 = np.squeeze(extra2[0, ...].data.cpu().numpy())
        extra2 = mask_vis(extra2)
        extra2 = np.array(Image.fromarray(extra2).convert("RGB"))
        img_list.append(extra2)
    if "gt" in ele:
        img_list.append(preprocess_depth(ele["gt"][0, ...]))

    img_merge = np.hstack(img_list)
    return img_merge.astype("uint8")


def add_row(img_merge, row):
    return np.vstack([img_merge, row])


def save_image_torch(rgb, filename):
    # torch2numpy
    rgb = validcrop(rgb)
    rgb = np.squeeze(rgb[0, ...].data.cpu().numpy())
    # print(rgb.size())
    rgb = np.transpose(rgb, (1, 2, 0))
    rgb = rgb.astype("uint8")
    image_to_write = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, image_to_write)


def save_depth_as_uint16png(img, filename):
    # from tensor
    img = np.squeeze(img.data.cpu().numpy())
    img = (img * 256).astype("uint16")
    cv2.imwrite(filename, img)


def save_depth_as_uint16png_upload(img, filename):
    # from tensor
    img = np.squeeze(img.data.cpu().numpy())
    img = (img * 256.0).astype("uint16")
    img_buffer = img.tobytes()
    imgsave = Image.new("I", img.T.shape)
    imgsave.frombytes(img_buffer, "raw", "I;16")
    imgsave.save(filename)


def save_depth_as_uint8colored(img, filename):
    # from tensor
    img = validcrop(img)
    img = np.squeeze(img.data.cpu().numpy())
    img = depth_colorize(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, img)


def save_mask_as_uint8colored(img, filename, colored=True, normalized=True):
    img = validcrop(img)
    img = np.squeeze(img.data.cpu().numpy())
    if normalized == False:
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
    if colored == True:
        img = 255 * cmap(img)[:, :, :3]
    else:
        img = 255 * img
    img = img.astype("uint8")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, img)


def save_feature_as_uint8colored(img, filename):
    img = validcrop(img)
    img = np.squeeze(img.data.cpu().numpy())
    img = feature_colorize(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, img)


cm = plt.get_cmap("plasma")


def depth_to_image(dep, max_d=10.0):
    dep = dep.detach()
    dep = dep[0, 0, :, :].data.cpu().numpy() / max_d
    dep = (255.0 * cm(dep)).astype("uint8")
    return dep[..., :3]


def rgb_to_image(rgb, img_mean, img_std):
    rgb = rgb.detach()
    rgb.mul_(img_std.type_as(rgb)).add_(img_mean.type_as(rgb))
    rgb = rgb[0, :, :, :].data.cpu().numpy()
    rgb = 255.0 * np.transpose(rgb, (1, 2, 0))
    rgb = np.clip(rgb, 0, 256).astype("uint8")
    return rgb


def save_image(img, path):
    img = Image.fromarray(img[:, :, :3], "RGB")
    img.save(path)


def offset_to_image(offset):
    offset = offset[0, 0, :, :].data.cpu().numpy()
    offset = (255.0 * cm(offset)).astype("uint8")
    return offset[..., :3]


def merge_into_row(rgb=None, dep=None, pred=None, gt=None, offset=None, dataset="nyu"):
    if dataset in ["nyu", "sunrgbd", "sparsity"]:
        max_depth = 10.0
        img_mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        img_std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)

    mask = gt > 0
    max_depth = gt.max().item()
    dep = depth_to_image(dep, max_depth)
    pred = depth_to_image(pred, max_depth)
    gt = depth_to_image(gt, max_depth)
    rgb = rgb_to_image(rgb, img_mean, img_std)
    offset = offset_to_image(offset * mask * 5)

    img = np.hstack([rgb, dep, pred, offset, gt])
    return img


def batch_save(rgb, dep, pred, gt, offset, path, dataset="nyu"):
    B = dep.shape[0]
    imgs = []
    for i in range(B):
        imgs.append(
            merge_into_row(
                rgb[i : i + 1, ...],
                dep[i : i + 1, ...],
                pred[i : i + 1, ...],
                gt[i : i + 1, ...],
                offset[i : i + 1, ...],
                dataset=dataset,
            )
        )
    imgs = np.vstack(imgs)
    save_image(imgs, path)


def batch_save_kitti(rgb, dep, pred, gt, offset, path):
    def preprocess_depth(x, max_d):
        cm = plt.get_cmap("plasma")
        y = np.squeeze((x / max_d).data.cpu().numpy())
        y = 255 * cm(y)[:, :, :3]
        # y = np.squeeze(x.data.cpu().numpy())
        return y.astype("uint8")

    B = dep.shape[0]
    imgs = []
    for i in range(B):
        mask = (gt[i] > 0.001).detach()
        r = rgb[i].data.cpu().numpy()
        r = np.transpose(r, (1, 2, 0))
        max_d = gt[i].max()
        d = preprocess_depth(dep[i], max_d)
        predd = preprocess_depth(pred[i], max_d)
        off = preprocess_depth(offset[i] * mask, max_d)
        gtd = preprocess_depth(gt[i], max_d)
        merge_imags = [r, d, predd, off, gtd]
        merge_imags = np.hstack(merge_imags)
        merge_imags = merge_imags.astype("uint8")
        imgs.append(merge_imags)
    imgs = np.vstack(imgs)
    image_to_write = cv2.cvtColor(imgs, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image_to_write)


def padding_kitti(pred, h_cropped):
    pred = torch.nn.functional.pad(pred, (0, 0, h_cropped, 0))
    crop = pred[:, :, h_cropped].unsqueeze(-2)
    pred[:, :, :h_cropped, :] = crop.repeat(1, 1, h_cropped, 1)
    return pred
