import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn import functional as F


class BayesianLinear(nn.Module):
    """
    Bayesian type of linear network.
    similar with Linear module of torch.nn.modules.
    
    There are 2 options for Bayesian Linear, Fully Factorized Gaussian(FFG) and Matrix Variate Gaussian.
    If you pick FFG option, 
    """
    def __init__(self, in_features, out_features, option='FFG', bias=True, eps=1e-8, lamb=1):
        if option not in ('FFG', 'MVG'):
            raise RuntimeError("Option should be FFG or MVG.")
        self.option = option
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        self.bias = Parameter(torch.FloatTensor(out_features))
        if self.option == 'FFG':
            # u : mean / f : variance
            self.u_weight = torch.FloatTensor(torch.FloatTensor(out_features, in_features))
            self.f_weight = torch.FloatTensor(torch.FloatTensor(out_features, in_features))
            self.u_bias = torch.FloatTensor(torch.FloatTensor(out_features))
            self.f_bias = torch.FloatTensor(torch.FloatTensor(out_features))
        else:
            pass
        super(BayesianLinear, self).__init__()

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def sample_parameters(self):
        if self.option == 'FFG':
            self.weight.data = self.u_weight + self.f_weight.sqrt() * torch.randn(self.f_weight.size())
            self.bias.data = self.u_bias + self.f_bias.sqrt() * torch.randn(self.f_bias.size())
        else:
            pass

    def update_bayesian_parameters(self, u_delta_dict, f_dict):
        """
        call after step function of Noisy Adam or Noisy KFAC.
        update bayesian parameters by given params.
        :return: 
        """
        self.u_weight += u_delta_dict[self.weight]
        self.u_bias += u_delta_dict[self.bias]
        self.f_weight = f_dict[self.weight]
        self.f_bias = f_dict[self.bias]


class BayesianMultilayer(nn.Module):
    def __init__(self, option='FFG'):
        self.model = nn.Sequential(
            BayesianLinear(
                in_features=784,
                out_features=100,
                option=option
            ),
            nn.ReLU(),
            BayesianLinear(
                in_features=100,
                out_features=50,
                option=option
            ),
            nn.ReLU(),
            BayesianLinear(
                in_features=50,
                out_features=10,
                option=option
            ),
            nn.ReLU()
        )
        self.u_delta_dict = None
        self.f_dict = None
        super(BayesianMultilayer, self).__init__()

    def forward(self, x):
        self.sample_parameters()
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

    def sample_parameters(self):
        self.model.apply(self._sample_parameters)

    def update_bayesian_parameters(self, u_delta_dict, f_dict):
        self.u_delta_dict = u_delta_dict
        self.f_dict = f_dict
        self.model.apply(self._update_bayesian_parameters)
