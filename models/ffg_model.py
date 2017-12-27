from collections import OrderedDict

import torch
from torch.autograd import Variable
from torch import nn

from models.base_model import BaseModel
from models.networks import BayesianMultilayer
from optimizers.noisy_adam import NoisyAdam


class FFGModel(BaseModel):
    def __init__(self, opt):
        super(FFGModel, self).__init__(opt)

        self.gpu_ids = opt.gpu_ids
        self.batch_size = opt.batch_size

        self.model = BayesianMultilayer(option=opt.option)
        self.model_optimizer = NoisyAdam(
            [self.model]
        )

        self.result = None

        self.input = self.Tensor(
            opt.batch_size,
            opt.initial_size
        )
        self.label = self.LabelTensor(
            opt.batch_size,
            opt.label_size
        )
        self.loss_function = nn.CrossEntropyLoss()
        self.loss = None

    @property
    def name(self):
        return 'FFGModel'

    def forward(self, volatile=False):
        self.result = self.model(self.input)

    def set_input(self, data):
        self.input.resize(data.size()).copy(data)

    def get_losses(self):
        raise OrderedDict([
            ('loss', self.loss.cpu().data.numpy()[0])
        ])

    def get_visuals(self, sample_single_image=True):
        raise NotImplemented

    def save(self, epoch):
        raise NotImplemented

    def remove(self, epoch):
        raise NotImplemented

    def load(self, epoch):
        raise NotImplemented

    def backward(self):
        self.loss = self.loss_function(self.result, self.label)
        self.loss.backward()

    def optimize_parameters(self):
        self.forward()

        self.model_optimizer.zero_grad()
        self.backward()
        self.model_optimizer.step()

        # update bayesian params here.
        u_delta_dict, f_dict = self.model_optimizer.get_delta_dicts()
        self.update_bayesian_parameters(u_delta_dict, f_dict)

    def update_bayesian_parameters(self, u_delta_dict, f_dict):
        self.model.update_bayesian_parameters(u_delta_dict, f_dict)

    def test(self):
        pass
