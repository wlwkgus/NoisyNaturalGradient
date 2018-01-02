import math
import torch
from torch.optim.optimizer import Optimizer
from itertools import chain

from models.constants import COMPLEX_BAYESIAN_NETWORKS, BASIC_BAYESIAN_NETWORKS


class NoisyKFAC(Optimizer):
    """
    Implements Noisy Adam algorithm.

    NoisyKFAC optimizer only works with matrix variate gaussian Bayesian Linear module in networks.py.


    It has been proposed in `Noisy Natural Gradient as Variational Inference`
    https://arxiv.org/pdf/1712.02390.pdf

    """

    @staticmethod
    def check_bayesian_and_option(network):
        name = network.__class__.__name__
        if 'Sequential' in name:
            return
        for network_name in COMPLEX_BAYESIAN_NETWORKS:
            if network_name in name:
                return
        for network_name in BASIC_BAYESIAN_NETWORKS:
            if network_name not in name and len(list(network.parameters())) > 0:
                raise RuntimeError("This implementation only supports BayesianLinear module.")
            elif network_name in name and network.option != 'MVG':
                raise RuntimeError("Noisy KFAC optimizer only supports matrix variate gaussian option.")

    def _save_input(self, network, i):
        if self.steps % self.t_stats == 0:
            aa = torch.mm(i[0].data.t(), i[0].data) / i[0].size(1)
            for p in network.parameters():
                self.aa_mappings[p] = aa

    def _save_grad_output(self, network, grad_input, grad_output):
        if self.steps % self.t_stats == 0:
            gg = torch.mm(grad_output[0].data.t(), grad_output[0].data) / grad_output[0].size(1)
            for p in network.parameters():
                self.gg_mappings[p] = gg

    def _register_hook(self, network):
        name = network.__class__.__name__
        if name in BASIC_BAYESIAN_NETWORKS:
            network.register_forward_pre_hook(self._save_input)
            network.register_backward_hook(self._save_grad_output)

    def __init__(self, networks, t_stats, t_inv, lr=1e-3, beta=0.99, eps=1e-2, weight_decay=0, phi=1):
        # TODO : support Bayesian Convolution.
        self.t_stats = t_stats
        self.t_inv = t_inv
        self.steps = 0
        self.aa_mappings = dict()
        self.gg_mappings = dict()
        for network in networks:
            network.apply(self.check_bayesian_and_option)
            network.apply(self._register_hook)
        defaults = dict(
            lr=lr,
            beta=beta,
            eps=eps,
            weight_decay=weight_decay,
            t_stats=t_stats,
            t_inv=t_inv,
            phi=phi,
        )
        params = chain(*[network.parameters() for network in networks])
        super(NoisyKFAC, self).__init__(params, defaults)

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
                    # KFAC
                    # TODO : data shape
                    # TODO : This implementation only supports linear module which has 2 dimension parameter.
                    # TODO : Need to modify to support 3 dimension parameter like Conv2d.
                    # A : moving exp of covariance matrix of input activation
                    # shape of A : input_dimension x input_dimension

                    # S : moving exp of covariance matrix of gradient of pre activation
                    # shape of S : output_dimension x output_dimension

                    if len(p.size()) == 2:
                        state['a'] = torch.zeros(p.size(1), p.size(1))
                        state['s'] = torch.zeros(p.size(0), p.size(0))
                        state['a_inverse'] = torch.zeros_like(state['a'])
                        state['s_inverse'] = torch.zeros_like(state['s'])
                    else:
                        raise Exception("Only linear without bias network available.")

                beta = group['beta']
                t_stats, t_inv = group['t_stats'], group['t_inv']

                state['step'] += 1
                self.steps = state['step']

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                if state['step'] % t_stats == 0:
                    state['a'].mul_(beta).add_(1 - beta, self.aa_mappings[p].cpu())
                    state['s'].mul_(beta).add_(1 - beta, self.gg_mappings[p].cpu())
                    # state['s'] = (1 - beta) * state['s'] + beta * self.gg_mappings[p]

                if state['step'] % t_inv == 0:
                    state['s_inverse'] = (
                        state['s'] + 1 / group['phi'] * math.sqrt(group['eps']) * torch.eye(state['s'].size(0))
                    ).inverse()
                    state['a_inverse'] = (
                        state['a'] + group['phi'] * math.sqrt(group['eps']) * torch.eye(state['a'].size(0))
                    ).inverse()

                step_size = group['lr'] * torch.mm(torch.mm(state['s_inverse'], grad), state['a_inverse'])

                p.data = -group['lr'] * step_size

        return loss

    def get_delta_dicts(self):
        u_delta_dict = dict()
        f_dict = dict()
        for group in self.param_groups:
            for p in group['params']:
                u_delta_dict[p] = p.data
                f_dict[p] = dict(
                    a_inverse=self.state[p]['a_inverse'],
                    s_inverse=self.state[p]['s_inverse'],
                )
        return u_delta_dict, f_dict
