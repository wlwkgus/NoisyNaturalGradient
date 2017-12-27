from torch import nn
from PIL import Image
import os
import numpy as np

"""
These functions are originated from junyanz's repository.
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git
"""


def layer_wrapper(
        layer,
        norm_layer=nn.BatchNorm2d,
        dropout_rate=0.0,
        activation_function=nn.LeakyReLU(negative_slope=0.1)
):
    layers = list()
    layers.append(layer)
    if norm_layer:
        if type(layers[-1]) == nn.Conv2d:
            layers.append(norm_layer(layers[-1].out_channels))
        elif type(layers[-1]) == nn.ConvTranspose2d:
            layers.append(norm_layer(layers[-1].out_channels))
        else:
            layers.append(norm_layer(layers[-1].out_channels))
    if dropout_rate > 0.0:
        layers.append(nn.Dropout(dropout_rate))
    if activation_function:
        layers.append(activation_function)

    return nn.Sequential(*layers)


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def tensor2im(image_tensor, imtype=np.uint8, sample_single_image=True):
    if sample_single_image:
        image_numpy = image_tensor[0].cpu().float().numpy()
        if image_numpy.shape[0] == 1:
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        return image_numpy.astype(imtype)
    else:
        image_numpy = image_tensor.cpu().float().numpy()
        if image_numpy.shape[1] == 1:
            image_numpy = np.tile(image_numpy, (1, 3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
        return image_numpy.astype(imtype)
