from collections import OrderedDict

import torch
from torch.autograd import Variable
from torch import nn

from models.base_model import BaseModel
from optimizers.noisy_adam import NoisyAdam


class MVGModel(BaseModel):
    def __init__(self, opt):
        super(MVGModel, self).__init__(opt)

        self.gpu_ids = opt.gpu_ids
        self.batch_size = opt.batch_size

        self.model = None
        self.model_optimizer = NoisyAdam(
            [self.model]
        )

    @property
    def name(self):
        return 'FFGModel'

    def forward(self, volatile=False):
        pass

    def set_input(self, data):
        pass

    def get_losses(self):
        raise NotImplemented

    def get_visuals(self, sample_single_image=True):
        raise NotImplemented

    def save(self, epoch):
        raise NotImplemented

    def remove(self, epoch):
        raise NotImplemented

    def load(self, epoch):
        raise NotImplemented

    def optimize_parameters(self):
        pass

    def test(self):
        pass
