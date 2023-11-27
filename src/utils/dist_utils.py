import torch.distributed as dist
import torch


def reduce_value(value, average=True):
    if not (torch.distributed.is_available()
            and torch.distributed.is_initialized()):
        return value
    world_size = dist.get_world_size()
    if world_size < 2:  # 单GPU的情况
        return value

    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value = value / world_size

        return value


def get_dist_info():
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def is_master():
    rank, _ = get_dist_info()
    if rank == 0:
        return True
    else:
        return False
