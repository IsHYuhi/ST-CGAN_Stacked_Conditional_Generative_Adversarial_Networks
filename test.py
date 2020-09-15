from utils.data_loader import make_datapath_list, ImageDataset, ImageTransform, ImageTransformOwn
from models.ST_CGAN import Generator, Discriminator
from torchvision.utils import make_grid
from torchvision.utils import save_image
from torchvision import models
from torchvision import transforms
from torch.autograd import Variable
from collections import OrderedDict
from PIL import Image
from tqdm import tqdm

import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import numpy as np
import argparse
import time
import torch
import os

torch.manual_seed(44)
# choose your device
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def get_parser():
    parser = argparse.ArgumentParser(
        prog='ST-CGAN: Stacked Conditional Generative Adversarial Networks for Jointly Learning Shadow Detection and Shadow Removal',
        usage='python3 main.py',
        description='This module demonstrates shadow detection and removal using ST-CGAN.',
        add_help=True)

    parser.add_argument('-l', '--load', type=str, default=None, help='the number of checkpoints')
    parser.add_argument('-i', '--image_path', type=str, default=None, help='file path of image you want to test')
    parser.add_argument('-o', '--out_path', type=str, default='./test_result', help='saving path')
    parser.add_argument('-s', '--image_size', type=int, default=286)
    parser.add_argument('-cs', '--crop_size', type=int, default=256)

    return parser

def fix_model_state_dict(state_dict):
    '''
    remove 'module.' of dataparallel
    '''
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]
        new_state_dict[name] = v
    return new_state_dict

def check_dir():
    if not os.path.exists('./test_result'):
        os.mkdir('./test_result')
    if not os.path.exists('./test_result/detected_shadow'):
        os.mkdir('./test_result/detected_shadow')
    if not os.path.exists('./test_result/shadow_removal_image'):
        os.mkdir('./test_result/shadow_removal_image')
    if not os.path.exists('./test_result/grid'):
        os.mkdir('./test_result/grid')

def unnormalize(x):
    x = x.transpose(1, 3)
    #mean, std
    x = x * torch.Tensor((0.5, )) + torch.Tensor((0.5, ))
    x = x.transpose(1, 3)
    return x

def test(G1, G2, test_dataset):
    '''
    this module test dataset from ISTD dataset
    '''
    check_dir()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    G1.to(device)
    G2.to(device)

    """use GPU in parallel"""
    if device == 'cuda':
        G1 = torch.nn.DataParallel(G1)
        G2 = torch.nn.DataParallel(G2)
        print("parallel mode")

    print("device:{}".format(device))

    G1.eval()
    G2.eval()
    
    for n, (img, gt_shadow, gt) in enumerate([test_dataset[i] for i in range(test_dataset.__len__())]):

        print(test_dataset.img_list['path_A'][n].split('/')[4][:-4])

        img = torch.unsqueeze(img, dim=0)
        gt_shadow = torch.unsqueeze(gt_shadow, dim=0)
        gt = torch.unsqueeze(gt, dim=0)

        with torch.no_grad():
            detected_shadow = G1(img.to(device))
            detected_shadow = detected_shadow.to(torch.device('cpu'))
            concat = torch.cat([img, detected_shadow], dim=1)
            shadow_removal_image = G2(concat.to(device))
            shadow_removal_image = shadow_removal_image.to(torch.device('cpu'))

        
        grid = make_grid(torch.cat([unnormalize(img), unnormalize(gt), unnormalize(shadow_removal_image),
                                    unnormalize(torch.cat([gt_shadow, gt_shadow, gt_shadow], dim=1)),
                                    unnormalize(torch.cat([detected_shadow, detected_shadow, detected_shadow], dim=1))],
                                    dim=0))

        save_image(grid, './test_result/grid/'+test_dataset.img_list['path_A'][n].split('/')[4])

        detected_shadow = transforms.ToPILImage(mode='L')(unnormalize(detected_shadow)[0, :, :, :])
        detected_shadow.save('./test_result/detected_shadow/'+test_dataset.img_list['path_A'][n].split('/')[4])

        shadow_removal_image = transforms.ToPILImage(mode='RGB')(unnormalize(shadow_removal_image)[0, :, :, :])
        shadow_removal_image.save('./test_result/shadow_removal_image/'+test_dataset.img_list['path_A'][n].split('/')[4])

def test_own_image(G1, G2, path, out_path, size, img_transform):
    img = Image.open(path).convert('RGB')
    width, height = img.width, img.height
    img = img.resize((size, size), Image.LANCZOS)
    img = img_transform(img)
    img = torch.unsqueeze(img, dim=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    G1.to(device)
    G2.to(device)

    """use GPU in parallel"""
    if device == 'cuda':
        G1 = torch.nn.DataParallel(G1)
        G2 = torch.nn.DataParallel(G2)
        print("parallel mode")

    print("device:{}".format(device))

    G1.eval()
    G2.eval()

    with torch.no_grad():
        detected_shadow = G1(img.to(device))
        detected_shadow = detected_shadow.to(torch.device('cpu'))
        concat = torch.cat([img, detected_shadow], dim=1)
        shadow_removal_image = G2(concat.to(device))
        shadow_removal_image = shadow_removal_image.to(torch.device('cpu'))

        
        grid = make_grid(torch.cat([unnormalize(img),
                                    unnormalize(torch.cat([detected_shadow, detected_shadow, detected_shadow], dim=1)),
                                     unnormalize(shadow_removal_image)],
                                    dim=0))

        save_image(grid, out_path + '/grid_' + path.split('/')[-1])

        detected_shadow = transforms.ToPILImage(mode='L')(unnormalize(detected_shadow)[0, :, :, :])
        detected_shadow= detected_shadow.resize((width, height), Image.LANCZOS)
        detected_shadow.save(out_path + '/detected_shadow_' + path.split('/')[-1])

        shadow_removal_image = transforms.ToPILImage(mode='RGB')(unnormalize(shadow_removal_image)[0, :, :, :])
        shadow_removal_image = shadow_removal_image.resize((width, height), Image.LANCZOS)
        shadow_removal_image.save(out_path + '/shadow_removal_image_' + path.split('/')[-1])

def main(parser):
    G1 = Generator(input_channels=3, output_channels=1)
    G2 = Generator(input_channels=4, output_channels=3)

    '''load'''
    if parser.load is not None:
        print('load checkpoint ' + parser.load)

        G1_weights = torch.load('./checkpoints/ST-CGAN_G1_'+parser.load+'.pth')
        G1.load_state_dict(fix_model_state_dict(G1_weights))

        G2_weights = torch.load('./checkpoints/ST-CGAN_G2_'+parser.load+'.pth')
        G2.load_state_dict(fix_model_state_dict(G2_weights))

    mean = (0.5,)
    std = (0.5,)
    size = parser.image_size
    crop_size = parser.crop_size

    # test own image
    if parser.image_path is not None:
        print('test ' + parser.image_path)
        test_own_image(G1, G2, parser.image_path, parser.out_path, size, img_transform=ImageTransformOwn(size=size, mean=mean, std=std))

    # test images from the ISTD dataset
    else:
        print('test ISTD dataset')
        test_img_list = make_datapath_list(phase='test')
        test_dataset = ImageDataset(img_list=test_img_list,
                                    img_transform=ImageTransform(size=size, crop_size=crop_size, mean=mean, std=std),
                                    phase='test')
        test(G1, G2, test_dataset)

if __name__ == "__main__":
    parser = get_parser().parse_args()
    main(parser)
