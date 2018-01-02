import math
import torch
from torch.optim.optimizer import Optimizer
from itertools import chain

from models.constants import COMPLEX_BAYESIAN_NETWORKS, BASIC_BAYESIAN_NETWORKS


class NoisyAdam(Optimizer):
    """
    Implements Noisy Adam algorithm.
    
    NoisyAdam optimizer only works with fully factorized gaussian Bayesian Linear module in networks.py.
    
     
    It has been proposed in `Noisy Natural Gradient as Variational Inference`
    https://arxiv.org/pdf/1712.02390.pdf
    
    
    """

    @staticmethod
    def check_bayesian_and_option(network):
        name = network.__class__.__name__
        # TODO : replace this hard-coded area
        if 'Sequential' in name:
            return
        for network_name in COMPLEX_BAYESIAN_NETWORKS:
            if network_name in name:
                return
        for network_name in BASIC_BAYESIAN_NETWORKS:
            if network_name not in name and len(list(network.parameters())) > 0:
                raise RuntimeError("This implementation only supports BayesianLinear module.")
            elif network_name in name and network.option != 'FFG':
                raise RuntimeError("Noisy Adam optimizer only supports fully factorized gaussian option.")

    def __init__(self, networks, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        # TODO : supports Bayesian Convolution?
        for network in networks:
            network.apply(self.check_bayesian_and_option)
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        params = chain(*[network.parameters() for network in networks])
        super(NoisyAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        """
        Performs a single optimization step.
        
        replace p.data with gradient delta value.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
                
        Return:
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                # This is different part between Adam and Noisy Adam.
                grad = p.grad.data - group['eps'] * p.data
                if grad.is_sparse:
                    raise RuntimeError('Does not support sparse gradients.')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data = torch.addcdiv(torch.zeros(1), -step_size, exp_avg, denom)

        return loss

    def get_delta_dicts(self):
        u_delta_dict = dict()
        f_dict = dict()
        for group in self.param_groups:
            for p in group['params']:
                u_delta_dict[p] = p.data
                f_dict[p] = self.state[p]['exp_avg_sq']
        return u_delta_dict, f_dict
