import torch
import os
import json
from glob import glob
from abc import abstractmethod


class BaseModel(object):
    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.is_train = opt.is_train
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.FloatTensor
        self.LabelTensor = torch.cuda.LongTensor if self.gpu_ids else torch.LongTensor
        self.save_dir = os.path.join(opt.ckpt_dir, opt.model)
        self.lr_scheduler = None

    @property
    def name(self):
        return 'BaseModel'

    @abstractmethod
    def forward(self, volatile=False):
        raise NotImplemented

    @abstractmethod
    def test(self):
        raise NotImplemented

    @abstractmethod
    def set_input(self, data):
        raise NotImplemented

    @abstractmethod
    def optimize_parameters(self):
        raise NotImplemented

    @abstractmethod
    def get_losses(self):
        raise NotImplemented

    @abstractmethod
    def get_visuals(self, sample_single_image=True):
        raise NotImplemented

    @abstractmethod
    def save(self, epoch):
        raise NotImplemented

    @abstractmethod
    def remove(self, epoch):
        raise NotImplemented

    @abstractmethod
    def load(self, epoch):
        raise NotImplemented

    def save_network(self, network, network_name, epoch_count, gpu_ids):
        save_filename = network_name + '-' + str(epoch_count)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)

        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(device=gpu_ids[0])

    def load_network(self, network, network_name, epoch_count):
        save_filename = network_name + '-' + str(epoch_count)
        save_path = os.path.join(self.save_dir, save_filename)
        network.load_state_dict(torch.load(save_path))

    def remove_checkpoint(self, network_name, epoch_count):
        save_filename = network_name + '-' + str(epoch_count)
        save_path = os.path.join(self.save_dir, save_filename)
        os.remove(save_path)
