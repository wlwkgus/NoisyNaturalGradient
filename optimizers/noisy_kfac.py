import math
import torch
from torch.optim.optimizer import Optimizer


class NoisyKFAC(Optimizer):
    """
    Implements Noisy Adam algorithm.

    It has been proposed in `Noisy Natural Gradient as Variational Inference`
    https://arxiv.org/pdf/1712.02390.pdf


    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(NoisyKFAC, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("This implementation does not support sparse gradients.")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0

                    state['exp_avg'] = torch.zeros_like(p.data)
