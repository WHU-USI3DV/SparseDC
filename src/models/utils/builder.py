import inspect
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.instancenorm import _InstanceNorm

conv_cfg = {
    'Conv': nn.Conv2d,
    'Conv1d': nn.Conv1d,
    'Conv2d': nn.Conv2d,
    'Conv3d': nn.Conv3d,
}

norm_layers = {
    'BN': nn.BatchNorm2d,
    'BN1d': nn.BatchNorm1d,
    'BN2d': nn.BatchNorm2d,
    'BN3d': nn.BatchNorm3d,
    'GN': nn.GroupNorm,
    'LN': nn.LayerNorm,
    'IN': nn.InstanceNorm2d,
    'IN1d': nn.InstanceNorm1d,
    'IN2d': nn.InstanceNorm2d,
    'IN3d': nn.InstanceNorm3d,
}

drop_layers = {
    'DropPath': nn.Dropout,
    'DropPath1d': nn.Dropout1d,
    'DropPath2d': nn.Dropout2d,
    'DropPath3d': nn.Dropout3d,
}

activation_layers = {
    'ReLU': nn.ReLU,
    'LeakyReLU': nn.LeakyReLU,
    'PReLU': nn.PReLU,
    'ReLU6': nn.ReLU6,
    'ELU': nn.ELU,
    'Tanh': nn.Tanh,
    'Sigmoid': nn.Sigmoid,
    'SiLU': nn.SiLU,
    'GELU': nn.GELU,
}

padding_layers = {
    'zero': nn.ZeroPad2d,
    'reflect': nn.ReflectionPad2d,
    'replicate': nn.ReflectionPad2d,
}


def infer_abbr(class_type):
    """Infer abbreviation from the class name.

    When we build a norm layer with `build_norm_layer()`, we want to preserve
    the norm type in variable names, e.g, self.bn1, self.gn. This method will
    infer the abbreviation to map class types to abbreviations.

    Rule 1: If the class has the property "_abbr_", return the property.
    Rule 2: If the parent class is _BatchNorm, GroupNorm, LayerNorm or
    InstanceNorm, the abbreviation of this layer will be "bn", "gn", "ln" and
    "in" respectively.
    Rule 3: If the class name contains "batch", "group", "layer" or "instance",
    the abbreviation of this layer will be "bn", "gn", "ln" and "in"
    respectively.
    Rule 4: Otherwise, the abbreviation falls back to "norm".

    Args:
        class_type (type): The norm layer type.

    Returns:
        str: The inferred abbreviation.
    """
    if not inspect.isclass(class_type):
        raise TypeError(
            f'class_type must be a type, but got {type(class_type)}')
    if hasattr(class_type, '_abbr_'):
        return class_type._abbr_
    if issubclass(class_type, _InstanceNorm):  # IN is a subclass of BN
        return 'in'
    elif issubclass(class_type, _BatchNorm):
        return 'bn'
    elif issubclass(class_type, nn.GroupNorm):
        return 'gn'
    elif issubclass(class_type, nn.LayerNorm):
        return 'ln'
    else:
        class_name = class_type.__name__.lower()
        if 'batch' in class_name:
            return 'bn'
        elif 'group' in class_name:
            return 'gn'
        elif 'layer' in class_name:
            return 'ln'
        elif 'instance' in class_name:
            return 'in'
        else:
            return 'norm_layer'


def build_conv_layer(cfg, *args, **kwargs):
    """Build convolution layer.
    Args:
        cfg (None or dict): Cfg should contain:
            type (str): Identify conv layer type.
            layer args: Args needed to instantiate a conv layer.
    Returns:
        nn.Module: Created conv layer.
    """
    if cfg is None:
        cfg_ = dict(type='Conv2d')
    else:
        if not isinstance(cfg, dict):
            raise TypeError('cfg must be a dict')
        if 'type' not in cfg:
            raise KeyError('the cfg dict must contain the key "type"')
        cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    conv_layer = conv_cfg[layer_type]

    layer = conv_layer(*args, **kwargs, **cfg_)

    return layer


def build_norm_layer(cfg, num_features: int, postfix=''):
    """Build normalization layer.

    Args:
        cfg (dict): The norm layer config, which should contain:

            - type (str): Layer type.
            - layer args: Args needed to instantiate a norm layer.
            - requires_grad (bool, optional): Whether stop gradient updates.
        num_features (int): Number of input channels.
        postfix (int | str): The postfix to be appended into norm abbreviation
            to create named layer.

    Returns:
        tuple[str, nn.Module]: The first element is the layer name consisting
        of abbreviation and postfix, e.g., bn1, gn. The second element is the
        created norm layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in norm_layers:
        raise KeyError(f'Unrecognized norm type {layer_type}')

    norm_layer = norm_layers[layer_type]
    abbr = infer_abbr(norm_layer)

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    requires_grad = cfg_.pop('requires_grad', True)
    cfg_.setdefault('eps', 1e-5)
    if layer_type != 'GN':
        layer = norm_layer(num_features, **cfg_)
    else:
        assert 'num_groups' in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)

    for param in layer.parameters():
        param.requires_grad = requires_grad

    return name, layer


def build_dropout(cfg):
    """Build convolution layer.
    Args:
        cfg (None or dict): Cfg should contain:
            type (str): Identify conv layer type.
            layer args: Args needed to instantiate a conv layer.
    Returns:
        nn.Module: Created conv layer.
    """
    layer_type = cfg.pop('type')
    drop_layer = drop_layers[layer_type]
    inplace = cfg.pop('inplace', False)

    layer = drop_layer(p=cfg['drop_prob'], inplace=inplace)

    return layer


def build_activation_layer(cfg):
    cfg_ = cfg.copy()
    layer_type = cfg_.pop('type')
    activation_layer = activation_layers[layer_type]

    layer = activation_layer(**cfg_)

    return layer


def build_padding_layer(cfg, *args, **kwargs):
    """Build padding layer.

    Args:
        cfg (dict): The padding layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate a padding layer.

    Returns:
        nn.Module: Created padding layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')

    cfg_ = cfg.copy()
    padding_type = cfg_.pop('type')
    if padding_type not in padding_layers:
        raise KeyError(f'Unrecognized padding type {padding_type}.')
    else:
        padding_layer = padding_layers.get(padding_type)

    layer = padding_layer(*args, **kwargs, **cfg_)

    return layer