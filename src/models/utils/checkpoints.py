import os
import os.path as osp
import pkgutil
from collections import OrderedDict
from importlib import import_module

import numpy as np
import torch
import torch.distributed as dist
import torchvision
from torch.nn import functional as F
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils import model_zoo

from src.utils.dist_utils import get_dist_info

open_mmlab_model_urls = {
    'vgg16_caffe': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/vgg16_caffe-292e1171.pth',  # noqa: E501
    'resnet50_caffe': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnet50_caffe-788b5fa3.pth',  # noqa: E501
    'resnet101_caffe': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnet101_caffe-3ad79236.pth',  # noqa: E501
    'resnext50_32x4d': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnext50-32x4d-0ab1a123.pth',  # noqa: E501
    'resnext101_32x4d': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnext101_32x4d-a5af3160.pth',  # noqa: E501
    'resnext101_64x4d': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnext101_64x4d-ee2c6f71.pth',  # noqa: E501
    'contrib/resnet50_gn': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnet50_gn_thangvubk-ad1730dd.pth',  # noqa: E501
    'detectron/resnet50_gn': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnet50_gn-9186a21c.pth',  # noqa: E501
    'detectron/resnet101_gn': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnet101_gn-cac0ab98.pth',  # noqa: E501
    'jhu/resnet50_gn_ws': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnet50_gn_ws-15beedd8.pth',  # noqa: E501
    'jhu/resnet101_gn_ws': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnet101_gn_ws-3e3c308c.pth',  # noqa: E501
    'jhu/resnext50_32x4d_gn_ws': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnext50_32x4d_gn_ws-0d87ac85.pth',  # noqa: E501
    'jhu/resnext101_32x4d_gn_ws': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnext101_32x4d_gn_ws-34ac1a9e.pth',  # noqa: E501
    'jhu/resnext50_32x4d_gn': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnext50_32x4d_gn-c7e8b754.pth',  # noqa: E501
    'jhu/resnext101_32x4d_gn': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnext101_32x4d_gn-ac3bb84e.pth',  # noqa: E501
    'msra/hrnetv2_w18': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/hrnetv2_w18-00eb2006.pth',  # noqa: E501
    'msra/hrnetv2_w32': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/hrnetv2_w32-dc9eeb4f.pth',  # noqa: E501
    'msra/hrnetv2_w40': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/hrnetv2_w40-ed0b031c.pth',  # noqa: E501
}  # yapf: disable



def load_checkpoint_adapter(model,
                            filename,
                            map_location=None,
                            strict=False,
                            logger=None):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Either a filepath or URL or modelzoo://xxxxxxx.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    # load checkpoint from modelzoo or file or url
    if filename.startswith('modelzoo://'):
        import torchvision
        model_urls = dict()
        for _, name, ispkg in pkgutil.walk_packages(
                torchvision.models.__path__):
            if not ispkg:
                _zoo = import_module('torchvision.models.{}'.format(name))
                if hasattr(_zoo, 'model_urls'):
                    _urls = getattr(_zoo, 'model_urls')
                    model_urls.update(_urls)
        model_name = filename[11:]
        checkpoint = model_zoo.load_url(model_urls[model_name])
    elif filename.startswith('open-mmlab://'):
        model_name = filename[13:]
        checkpoint = model_zoo.load_url(open_mmlab_model_urls[model_name])
    elif filename.startswith(('http://', 'https://')):
        checkpoint = model_zoo.load_url(filename)
    else:
        if not osp.isfile(filename):
            raise IOError('{} is not a checkpoint file'.format(filename))
        checkpoint = torch.load(filename, map_location=map_location)
    # get state_dict from checkpoint
    if isinstance(checkpoint, OrderedDict):
        state_dict = checkpoint
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']  # for classification weights
    else:
        state_dict = checkpoint
        # raise RuntimeError(
        #     'No state_dict found in checkpoint file {}'.format(filename))
    # strip prefix of state_dict
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}
    # load state_dict
    if hasattr(model, 'module'):
        load_state_dict(model.module, state_dict, strict, logger)
    else:
        load_state_dict(model, state_dict, strict, logger)
    return checkpoint


def get_torchvision_models():
    model_urls = dict()
    for _, name, ispkg in pkgutil.walk_packages(torchvision.models.__path__):
        if ispkg:
            continue
        _zoo = import_module(f'torchvision.models.{name}')
        if hasattr(_zoo, 'model_urls'):
            _urls = getattr(_zoo, 'model_urls')
            model_urls.update(_urls)
    return model_urls


def load_url_dist(url, model_dir=None):
    """In distributed setting, this function only download checkpoint at local
    rank 0."""
    rank, world_size = get_dist_info()
    rank = int(os.environ.get('LOCAL_RANK', rank))
    if rank == 0:
        checkpoint = model_zoo.load_url(url, model_dir=model_dir)
    if world_size > 1:
        torch.distributed.barrier()
        if rank > 0:
            checkpoint = model_zoo.load_url(url, model_dir=model_dir)
    return checkpoint


def is_module_wrapper(module):
    module_wrappers = (DataParallel, DistributedDataParallel)
    return isinstance(module, module_wrappers)


def load_state_dict(module, state_dict, strict=False, logger=None):
    """Load state_dict to a module.
    This method is modified from :meth:`torch.nn.Module.load_state_dict`.
    Default value for ``strict`` is set to ``False`` and the message for
    param mismatch will be shown even if strict is False.
    Args:
        module (Module): Module that receives the state_dict.
        state_dict (OrderedDict): Weights.
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``False``.
        logger (:obj:`logging.Logger`, optional): Logger to log the error
            message. If not specified, print function will be used.
    """
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    # use _load_from_state_dict to enable checkpoint version control
    def load(module, prefix=''):
        # recursively check parallel module in case that the model has a
        # complicated structure, e.g., nn.Module(nn.Module(DDP))
        if is_module_wrapper(module):
            module = module.module
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     all_missing_keys, unexpected_keys,
                                     err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
    load = None  # break load->load reference cycle

    # ignore "num_batches_tracked" of BN layers
    missing_keys = [
        key for key in all_missing_keys if 'num_batches_tracked' not in key
    ]

    if unexpected_keys:
        err_msg.append('unexpected key in source '
                       f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

    rank, _ = get_dist_info()
    if len(err_msg) > 0 and rank == 0:
        err_msg.insert(
            0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)


def _load_checkpoint(filename, map_location=None):
    """Load checkpoint from somewhere (modelzoo, file, url).
    Args:
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str | None): Same as :func:`torch.load`. Default: None.
    Returns:
        dict | OrderedDict: The loaded checkpoint. It can be either an
            OrderedDict storing model weights or a dict containing other
            information, which depends on the checkpoint.
    """
    if filename.startswith('modelzoo://'):
        model_urls = get_torchvision_models()
        model_name = filename[11:]
        checkpoint = load_url_dist(model_urls[model_name])
    elif filename.startswith(('http://', 'https://')):
        checkpoint = load_url_dist(filename)
    else:
        if not osp.isfile(filename):
            raise IOError(f'{filename} is not a checkpoint file')
        checkpoint = torch.load(filename, map_location=map_location)
    return checkpoint


def load_checkpoint(model,
                    filename,
                    map_location='cpu',
                    strict=False,
                    logger=None):
    """Load checkpoint from a file or URI.
    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.
    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    checkpoint = _load_checkpoint(filename, map_location)
    # OrderedDict is a subclass of dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f'No state_dict found in checkpoint file {filename}')
    # get state_dict from checkpoint
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    # strip prefix of state_dict
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    # for MoBY, load model of online branch
    if sorted(list(state_dict.keys()))[0].startswith('encoder'):
        state_dict = {
            k.replace('encoder.', ''): v
            for k, v in state_dict.items() if k.startswith('encoder.')
        }

    # reshape absolute position embedding
    if state_dict.get('absolute_pos_embed') is not None:
        absolute_pos_embed = state_dict['absolute_pos_embed']
        N1, L, C1 = absolute_pos_embed.size()
        N2, C2, H, W = model.absolute_pos_embed.size()
        if N1 != N2 or C1 != C2 or L != H * W:
            logger.warning("Error in loading absolute_pos_embed, pass")
        else:
            state_dict['absolute_pos_embed'] = absolute_pos_embed.view(
                N2, H, W, C2).permute(0, 3, 1, 2)

    # interpolate position bias table if needed
    relative_position_bias_table_keys = [
        k for k in state_dict.keys() if "relative_position_bias_table" in k
    ]
    for table_key in relative_position_bias_table_keys:
        table_pretrained = state_dict[table_key]
        table_current = model.state_dict()[table_key]
        L1, nH1 = table_pretrained.size()
        L2, nH2 = table_current.size()
        if nH1 != nH2:
            logger.warning(f"Error in loading {table_key}, pass")
        else:
            if L1 != L2:
                S1 = int(L1**0.5)
                S2 = int(L2**0.5)
                table_pretrained_resized = F.interpolate(
                    table_pretrained.permute(1, 0).view(1, nH1, S1, S1),
                    size=(S2, S2),
                    mode='bicubic')
                state_dict[table_key] = table_pretrained_resized.view(
                    nH2, L2).permute(1, 0)

    # load state_dict
    load_state_dict(model, state_dict, strict, logger)
    return checkpoint


def load_checkpoint_swin(model,
                         filename,
                         map_location='cpu',
                         strict=False,
                         rpe_interpolation='outer_mask'):
    """Load checkpoint from a file or URI.
    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.
    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    checkpoint = _load_checkpoint(filename, map_location)
    # OrderedDict is a subclass of dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f'No state_dict found in checkpoint file {filename}')
    # get state_dict from checkpoint
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'module' in checkpoint:
        state_dict = checkpoint['module']
    else:
        state_dict = checkpoint
    # strip prefix of state_dict
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    # for MoBY, load model of online branch
    if sorted(list(state_dict.keys()))[2].startswith('encoder'):
        state_dict = {
            k.replace('encoder.', ''): v
            for k, v in state_dict.items() if k.startswith('encoder.')
        }

    # reshape absolute position embedding for Swin
    if state_dict.get('absolute_pos_embed') is not None:
        absolute_pos_embed = state_dict['absolute_pos_embed']
        N1, L, C1 = absolute_pos_embed.size()
        N2, C2, H, W = model.absolute_pos_embed.size()
        if N1 != N2 or C1 != C2 or L != H * W:
            print("Error in loading absolute_pos_embed, pass")
        else:
            state_dict['absolute_pos_embed'] = absolute_pos_embed.view(
                N2, H, W, C2).permute(0, 3, 1, 2)

    # interpolate position bias table if needed
    relative_position_bias_table_keys = [
        k for k in state_dict.keys() if "relative_position_bias_table" in k
    ]
    for k in relative_position_bias_table_keys:
        table_pretrained = state_dict[k]
        table_current = model.state_dict()[k]
        L1, nH1 = table_pretrained.size()
        L2, nH2 = table_current.size()
        if nH1 != nH2:
            print(f"Error in loading {k}, pass")
        else:
            if L1 != L2:
                if rpe_interpolation in ['bicubic', 'bilinear', 'nearest']:
                    print(
                        f"Interpolate relative_position_bias_table using {rpe_interpolation}"
                    )
                    S1 = int(L1**0.5)
                    S2 = int(L2**0.5)
                    table_pretrained_resized = F.interpolate(
                        table_pretrained.permute(1, 0).view(1, nH1, S1, S1),
                        size=(S2, S2),
                        mode=rpe_interpolation)
                    state_dict[k] = table_pretrained_resized.view(nH2,
                                                                  L2).permute(
                                                                      1, 0)
                elif rpe_interpolation == 'geo':
                    print(
                        "Interpolate relative_position_bias_table using geo.")
                    src_size = int(L1**0.5)
                    dst_size = int(L2**0.5)

                    def geometric_progression(a, r, n):
                        return a * (1.0 - r**n) / (1.0 - r)

                    left, right = 1.01, 1.5
                    while right - left > 1e-6:
                        q = (left + right) / 2.0
                        gp = geometric_progression(1, q, src_size // 2)
                        if gp > dst_size // 2:
                            right = q
                        else:
                            left = q

                    # if q > 1.13492:
                    #     q = 1.13492

                    dis = []
                    cur = 1
                    for i in range(src_size // 2):
                        dis.append(cur)
                        cur += q**(i + 1)

                    r_ids = [-_ for _ in reversed(dis)]

                    x = r_ids + [0] + dis
                    y = r_ids + [0] + dis

                    t = dst_size // 2.0
                    dx = np.arange(-t, t + 0.1, 1.0)
                    dy = np.arange(-t, t + 0.1, 1.0)

                    all_rel_pos_bias = []

                    for i in range(nH1):
                        z = table_pretrained[:,
                                             i].view(src_size,
                                                     src_size).float().numpy()
                        f_cubic = F.interpolate.interp2d(x, y, z, kind='cubic')
                        all_rel_pos_bias.append(
                            torch.Tensor(f_cubic(dx, dy)).contiguous().view(
                                -1, 1).to(table_pretrained.device))

                    new_rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)
                    state_dict[k] = new_rel_pos_bias

    if 'pos_embed' in state_dict:
        pos_embed_checkpoint = state_dict['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int(
            (pos_embed_checkpoint.shape[-2] - num_extra_tokens)**0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches**0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            if dist.get_rank() == 0:
                print("Position interpolate from %dx%d to %dx%d" %
                      (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size,
                                            embedding_size).permute(
                                                0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(pos_tokens,
                                                         size=(new_size,
                                                               new_size),
                                                         mode='bicubic',
                                                         align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            state_dict['pos_embed'] = new_pos_embed

    # load state_dict
    load_state_dict(model, state_dict, strict)
    del checkpoint
    torch.cuda.empty_cache()
