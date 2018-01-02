import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import math


class BayesianLinear(nn.Module):
    """
    Bayesian type of linear network.
    similar with Linear module of torch.nn.modules.
    
    There are 2 options for Bayesian Linear, Fully Factorized Gaussian(FFG) and Matrix Variate Gaussian.
    If you pick FFG option, 
    """
    def __init__(self, in_features, out_features, n, gpu_ids, option='FFG', bias=True, eps=1e-8, lamb=1):
        super(BayesianLinear, self).__init__()
        if option not in ('FFG', 'MVG'):
            raise RuntimeError("Option should be FFG or MVG.")
        self.option = option
        self.gpu_ids = gpu_ids
        self.eps = eps
        self.lamb = lamb
        self.n = n
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        self.bias = None
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        if self.option == 'FFG':
            # u : mean / f : variance
            self.u_weight = torch.FloatTensor(torch.FloatTensor(out_features, in_features))
            self.f_weight = torch.FloatTensor(torch.FloatTensor(out_features, in_features))
            if self.bias is not None:
                self.u_bias = torch.FloatTensor(torch.FloatTensor(out_features))
                self.f_bias = torch.FloatTensor(torch.FloatTensor(out_features))
        elif self.option == 'MVG':
            self.u_weight = torch.FloatTensor(torch.FloatTensor(out_features, in_features))
            self.a_weight = torch.FloatTensor(torch.FloatTensor(in_features, in_features))
            self.s_weight = torch.FloatTensor(torch.FloatTensor(out_features, out_features))
            if self.bias is not None:
                raise Exception("MVG option does not support bias option.")
        self.reset_parameters()

    def reset_parameters(self):
        if self.option == 'FFG':
            stdv = 1. / math.sqrt(self.u_weight.size(1))
            self.u_weight.uniform_(-stdv, stdv)
            self.f_weight.fill_(0.)
            if self.bias is not None:
                stdv = 1. / math.sqrt(self.u_bias.size(0))
                self.u_bias.uniform_(-stdv, stdv)
                self.f_bias.fill_(0.)
        elif self.option == 'MVG':
            stdv = 1. / math.sqrt(self.u_weight.size(1))
            self.u_weight.uniform_(-stdv, stdv)
            self.a_weight.fill_(0.)
            self.s_weight.fill_(0.)

        else:
            raise Exception("option should be FFG or MVG.")

    def forward(self, x):
        if self.gpu_ids and isinstance(x.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(F.linear, (x, self.weight, self.bias), self.gpu_ids)
        return F.linear(x, self.weight, self.bias)

    def save_parameters(self):
        if self.option == 'FFG':
            self.weight.data = self.u_weight
            if self.bias is not None:
                self.bias.data = self.u_bias
        else:
            self.weight.data = self.u_weight
            if self.bias is not None:
                self.bias.data = self.u_bias

    def sample_parameters(self):
        if self.option == 'FFG':
            self.weight.data = self.u_weight + self.lamb /\
                                               (self.f_weight + self.eps).sqrt() * torch.randn(self.f_weight.size()) /\
                                               self.n
            if self.bias is not None:
                self.bias.data = self.u_bias + self.lamb / (self.f_bias + self.eps).sqrt() * torch.randn(
                    self.f_bias.size()) / self.n
        else:
            s_eval, s_evec = self.s_weight.symeig(eigenvectors=True)
            a_eval, a_evec = self.a_weight.symeig(eigenvectors=True)
            sqrt_s_weight = torch.mm(
                torch.mm(
                    s_evec, torch.diag(
                        s_eval.sign() * s_eval.abs().sqrt())
                ), s_evec.t()
            )
            sqrt_a_weight = torch.mm(
                torch.mm(
                    a_evec, torch.diag(
                        a_eval.sign() * a_eval.abs().sqrt())
                ), a_evec.t()
            )
            self.weight.data = self.u_weight + self.lamb / self.n *\
                                               torch.mm(
                                                   torch.mm(
                                                    sqrt_s_weight, torch.randn(self.u_weight.size())
                                                   ), sqrt_a_weight
                                               )

    def update_bayesian_parameters(self, u_delta_dict, f_dict):
        """
        call after step function of Noisy Adam or Noisy KFAC.
        update bayesian parameters by given params.
        :return: 
        """
        if self.option == 'FFG':
            self.u_weight += u_delta_dict[self.weight]
            self.f_weight = f_dict[self.weight]
            if self.bias is not None:
                self.u_bias += u_delta_dict[self.bias]
                self.f_bias = f_dict[self.bias]
        elif self.option == 'MVG':
            self.u_weight += u_delta_dict[self.weight]
            self.a_weight = f_dict[self.weight]['a_inverse']
            self.s_weight = f_dict[self.weight]['s_inverse']


class BayesianMultilayer(nn.Module):
    def __init__(self, gpu_ids, n, eps, option='FFG', bias=False):
        super(BayesianMultilayer, self).__init__()
        self.gpu_ids = gpu_ids
        self.n = n
        self.model = nn.Sequential(
            BayesianLinear(
                in_features=784,
                out_features=100,
                option=option,
                n=self.n,
                gpu_ids=self.gpu_ids,
                bias=bias,
                eps=eps,
            ),
            nn.ReLU(),
            BayesianLinear(
                in_features=100,
                out_features=10,
                option=option,
                n=self.n,
                gpu_ids=self.gpu_ids,
                bias=bias,
                eps=eps
            ),
            nn.ReLU()
        )
        self.u_delta_dict = None
        self.f_dict = None

    def forward(self, x, is_test=False):
        if not is_test:
            self.sample_parameters()
        if self.gpu_ids and isinstance(x.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, x, self.gpu_ids)
        return self.model(x)

    @staticmethod
    def _sample_parameters(m):
        classname = m.__class__.__name__
        if classname.find('BayesianLinear') != -1:
            m.sample_parameters()

    def _update_bayesian_parameters(self, m):
        classname = m.__class__.__name__
        if classname.find('BayesianLinear') != -1:
            m.update_bayesian_parameters(self.u_delta_dict, self.f_dict)

    def _save_parameters(self, m):
        classname = m.__class__.__name__
        if classname.find('BayesianLinear') != -1:
            m.save_parameters()

    def sample_parameters(self):
        self.model.apply(self._sample_parameters)

    def update_bayesian_parameters(self, u_delta_dict, f_dict):
        self.u_delta_dict = u_delta_dict
        self.f_dict = f_dict
        self.model.apply(self._update_bayesian_parameters)

    def save_parameters(self):
        self.model.apply(self._save_parameters)
