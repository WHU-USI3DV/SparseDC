import torch
from collections import defaultdict


class HybridOptim(torch.optim.Optimizer):
    # Wrapper around multiple optimizers that should be executed at the same time
    def __init__(self, optimizers):
        self.optimizers = optimizers

    @property
    def state(self):
        state = defaultdict(dict)
        for optimizer in self.optimizers:
            state = {**state, **optimizer.state}
        return state

    @property
    def param_groups(self):
        param_groups = []
        for optimizer in self.optimizers:
            param_groups = param_groups + optimizer.param_groups
        return param_groups

    def __getstate__(self):
        return [optimizer.__getstate__() for optimizer in self.optimizers]

    def __setstate__(self, state):
        for opt_state, optimizer in zip(self.optimizers, state):
            optimizer.__setstate__(opt_state)

    def __repr__(self):
        format_string = self.__class__.__name__ + ' ('
        for optimizer in self.optimizers:
            format_string += '\n'
            format_string += optimizer.__repr__()
        format_string += ')'
        return format_string

    def _hook_for_profile(self):
        for optimizer in self.optimizers:
            optimizer._hook_for_profile()

    def state_dict(self):
        return [optimizer.state_dict() for optimizer in self.optimizers]

    def load_state_dict(self, state_dict):
        for state, optimizer in zip(state_dict, self.optimizers):
            optimizer.load_state_dict(state)

    def zero_grad(self, set_to_none: bool = False):
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def add_param_group(self, param_group):
        raise NotImplementedError()

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for optimizer in self.optimizers:
            optimizer.step()

        return loss


class HybridLRS():
    """ Wrapper Class around lr_scheduler to return a 'dummy' optimizer to pass
        pytorch lightning checks
    """

    def __init__(self, hybrid_optimizer, idx, lr_scheduler) -> None:
        self.optimizer = hybrid_optimizer
        self.idx = idx
        self.lr_scheduler = lr_scheduler

    def __getattribute__(self, __name: str):
        if __name in {"optimizer", "idx", "lr_scheduler"}:
            return super().__getattribute__(__name)
        else:
            return self.lr_scheduler.__getattribute__(__name)
