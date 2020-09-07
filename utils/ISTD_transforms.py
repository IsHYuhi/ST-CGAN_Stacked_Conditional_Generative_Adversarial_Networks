#refered https://github.com/pytorch/vision/blob/master/torchvision/transforms/transforms.py

import math
import numbers
import random
import warnings
from collections.abc import Sequence
from typing import Tuple, List, Optional

import torch
from PIL import Image
from torch import Tensor
import torchvision.transforms.functional as F


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ToTensor(object):
    def __call__(self, pic):
        return F.to_tensor(pic[0]), F.to_tensor(pic[1]), F.to_tensor(pic[2])

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Scale(object):
  def __init__(self, size, interpolation=Image.BILINEAR):
    self.size = size
    self.interpolation = interpolation

  def __call__(self, imgs):
    output = []
    for img in imgs:
      w, h = img.size
      if (w <= h and w == self.size) or (h <= w and h == self.size):
        output.append(img)
        continue
      if w < h:
        ow = self.size
        oh = int(self.size * h / w)
        output.append(img.resize((ow, oh), self.interpolation))
        continue
      else:
        oh = self.size
        ow = int(self.size * w / h)
      output.append(img.resize((ow, oh), self.interpolation))
    return output[0], output[1], output[2]


class Normalize(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensor):
        return F.normalize(tensor[0], self.mean, self.std, self.inplace), F.normalize(tensor[1], self.mean, self.std, self.inplace), F.normalize(tensor[2], self.mean, self.std, self.inplace)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class CenterCrop(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        elif isinstance(size, Sequence) and len(size) == 1:
            self.size = (size[0], size[0])
        else:
            if len(size) != 2:
                raise ValueError("Please provide only two dimensions (h, w) for size.")

            self.size = size

    def forward(self, img):
        return F.center_crop(img[0], self.size), F.center_crop(img[1], self.size), F.center_crop(img[2], self.size)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class RandomCrop(torch.nn.Module):
    @staticmethod
    def get_params(img: Tensor, output_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = torch.randint(0, h - th + 1, size=(1, )).item()
        j = torch.randint(0, w - tw + 1, size=(1, )).item()
        return i, j, th, tw

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant"):
        super().__init__()
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        elif isinstance(size, Sequence) and len(size) == 1:
            self.size = (size[0], size[0])
        else:
            if len(size) != 2:
                raise ValueError("Please provide only two dimensions (h, w) for size.")

            # cast to tuple for torchscript
            self.size = tuple(size)
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def forward(self, img):
        if self.padding is not None:
            img[0] = F.pad(img[0], self.padding, self.fill, self.padding_mode)

        width, height = img[0].size
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img[0] = F.pad(img[0], padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img[0] = F.pad(img[0], padding, self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img[0], self.size)

        return F.crop(img[0], i, j, h, w), F.crop(img[1], i, j, h, w), F.crop(img[2], i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + "(size={0}, padding={1})".format(self.size, self.padding)


class RandomHorizontalFlip(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img):
        if torch.rand(1) < self.p:
            return F.hflip(img[0]), F.hflip(img[1]), F.hflip(img[2])
        return img[0], img[1], img[2]

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)