from utils.data_loader import make_datapath_list, ImageDataset, ImageTransform
from models.ST_CGAN import Generator, Discriminator
from torchvision.utils import make_grid
from torchvision.utils import save_image
from torch.autograd import Variable
from collections import OrderedDict
from torchvision import models
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

    parser.add_argument('-e', '--epoch', type=int, default=10000, help='Number of epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('-l', '--load', type=str, default=None, help='the number of checkpoints')
    parser.add_argument('-hor', '--hold_out_ratio', type=float, default=0.8, help='training-validation ratio')
    parser.add_argument('-s', '--image_size', type=int, default=286)
    parser.add_argument('-cs', '--crop_size', type=int, default=256)
    parser.add_argument('-lr', '--lr', type=float, default=2e-4)

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

def set_requires_grad(nets, requires_grad=False):
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def unnormalize(x):
    x = x.transpose(1, 3)
    #mean, std
    x = x * torch.Tensor((0.5, )) + torch.Tensor((0.5, ))
    x = x.transpose(1, 3)
    return x

def evaluate(G1, G2, dataset, device, filename):
    img, gt_shadow, gt = zip(*[dataset[i] for i in range(8)])
    img = torch.stack(img)
    gt_shadow = torch.stack(gt_shadow)
    gt = torch.stack(gt)

    with torch.no_grad():
        detected_shadow = G1(img.to(device))
        detected_shadow = detected_shadow.to(torch.device('cpu'))
        concat = torch.cat([img, detected_shadow], dim=1)
        shadow_removal_image = G2(concat.to(device))
        shadow_removal_image = shadow_removal_image.to(torch.device('cpu'))

    grid_detect = make_grid(torch.cat((unnormalize(gt_shadow), unnormalize(detected_shadow)), dim=0))
    grid_removal = make_grid(torch.cat((unnormalize(img), unnormalize(gt), unnormalize(shadow_removal_image)), dim=0))

    save_image(grid_detect, filename+'_detect.jpg')
    save_image(grid_removal, filename+'_removal.jpg')

def plot_log(data, save_model_name='model'):
    plt.cla()
    plt.plot(data['G'], label='G_loss ')
    plt.plot(data['D'], label='D_loss ')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss')
    plt.savefig('./logs/'+save_model_name+'.png')

def check_dir():
    if not os.path.exists('./logs'):
        os.mkdir('./logs')
    if not os.path.exists('./checkpoints'):
        os.mkdir('./checkpoints')
    if not os.path.exists('./result'):
        os.mkdir('./result')

def train_model(G1, G2, D1, D2, dataloader, val_dataset, num_epochs, parser, save_model_name='model'):

    check_dir()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    G1.to(device)
    G2.to(device)
    D1.to(device)
    D2.to(device)

    """use GPU in parallel"""
    if device == 'cuda':
        G1 = torch.nn.DataParallel(G1)
        G2 = torch.nn.DataParallel(G2)
        D1 = torch.nn.DataParallel(D1)
        D2 = torch.nn.DataParallel(D2)
        print("parallel mode")

    print("device:{}".format(device))

    lr = parser.lr
    beta1, beta2 = 0.5, 0.999

    optimizerG = torch.optim.Adam([{'params': G1.parameters()}, {'params': G2.parameters()}],
                                  lr=lr,
                                  betas=(beta1, beta2))
    optimizerD = torch.optim.Adam([{'params': D1.parameters()}, {'params': D2.parameters()}],
                                  lr=lr,
                                  betas=(beta1, beta2))

    criterionGAN = nn.BCEWithLogitsLoss().to(device)
    criterionL1 = nn.L1Loss().to(device)

    torch.backends.cudnn.benchmark = True

    mini_batch_size = parser.batch_size
    num_train_imgs = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    lambda_dict = {'lambda1':5, 'lambda2':0.1, 'lambda3':0.1}

    iteration = 1
    g_losses = []
    d_losses = []

    for epoch in range(num_epochs+1):

        G1.train()
        G2.train()
        D1.train()
        D2.train()
        t_epoch_start = time.time()

        epoch_g_loss = 0.0
        epoch_d_loss = 0.0

        print('-----------')
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('(train)')

        for images, gt_shadow, gt in tqdm(dataloader):

            # if size of minibatch is 1, an error would be occured.
            if images.size()[0] == 1:
                continue

            images = images.to(device)
            gt = gt.to(device)
            gt_shadow = gt_shadow.to(device)

            mini_batch_size = images.size()[0]

            # Train Discriminator
            set_requires_grad([D1, D2], True)  # enable backprop$
            optimizerD.zero_grad()

            # for D1
            detected_shadow = G1(images)
            fake1 = torch.cat([images, detected_shadow], dim=1)
            real1 = torch.cat([images, gt_shadow], dim=1)
            out_D1_fake = D1(fake1.detach())
            out_D1_real = D1(real1)# .detach() is not required as real1 doesn't have grad

            # for D2
            shadow_removal_image = G2(fake1)
            fake2 = torch.cat([fake1, shadow_removal_image], dim=1)
            real2 = torch.cat([real1, gt], dim=1)
            out_D2_fake = D2(fake2.detach())
            out_D2_real = D2(real2)# .detach() is not required as real2 doesn't have grad

            # L_CGAN1
            label_D1_fake = Variable(Tensor(np.zeros(out_D1_fake.size())), requires_grad=True)
            label_D1_real = Variable(Tensor(np.ones(out_D1_fake.size())), requires_grad=True)

            loss_D1_fake = criterionGAN(out_D1_fake, label_D1_fake)
            loss_D1_real = criterionGAN(out_D1_real, label_D1_real)
            D_L_CGAN1 = loss_D1_fake + loss_D1_real

            # L_CGAN2
            label_D2_fake = Variable(Tensor(np.zeros(out_D2_fake.size())), requires_grad=True)
            label_D2_real = Variable(Tensor(np.ones(out_D2_fake.size())), requires_grad=True)

            loss_D2_fake = criterionGAN(out_D2_fake, label_D2_fake)
            loss_D2_real = criterionGAN(out_D2_real, label_D2_real)
            D_L_CGAN2 = loss_D2_fake + loss_D2_real

            # total
            D_loss = lambda_dict['lambda2'] * D_L_CGAN1 + lambda_dict['lambda3'] * D_L_CGAN2
            D_loss.backward()
            optimizerD.step()

            # Train Generator
            set_requires_grad([D1, D2], False)
            optimizerG.zero_grad()

            # L_CGAN1
            fake1 = torch.cat([images, detected_shadow], dim=1)
            out_D1_fake = D1(fake1.detach())
            G_L_CGAN1 = criterionGAN(out_D1_fake, label_D1_real)

            # L_data1
            G_L_data1 = criterionL1(detected_shadow, gt_shadow)

            # L_CGAN2
            fake2 = torch.cat([fake1, shadow_removal_image], dim=1)
            out_D2_fake = D2(fake2.detach())
            G_L_CGAN2 = criterionGAN(out_D2_fake, label_D2_real)

            #L_data2
            G_L_data2 = criterionL1(gt, shadow_removal_image)

            #total
            G_loss = G_L_data1 + lambda_dict['lambda1'] * G_L_data2 + lambda_dict['lambda2'] * G_L_CGAN1 + lambda_dict['lambda3'] * G_L_CGAN2
            G_loss.backward()
            optimizerG.step()

            epoch_d_loss += D_loss.item()
            epoch_g_loss += G_loss.item()

        t_epoch_finish = time.time()
        print('-----------')
        print('epoch {} || Epoch_D_Loss:{:.4f} || Epoch_G_Loss:{:.4f}'.format(epoch, epoch_d_loss/batch_size, epoch_g_loss/batch_size))
        print('timer: {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))

        d_losses += [epoch_d_loss/batch_size]
        g_losses += [epoch_g_loss/batch_size]
        t_epoch_start = time.time()
        plot_log({'G':g_losses, 'D':d_losses}, save_model_name)

        if(epoch%10 == 0):
            torch.save(G1.state_dict(), 'checkpoints/'+save_model_name+'_G1_'+str(epoch)+'.pth')
            torch.save(G2.state_dict(), 'checkpoints/'+save_model_name+'_G2_'+str(epoch)+'.pth')
            torch.save(D1.state_dict(), 'checkpoints/'+save_model_name+'_D1_'+str(epoch)+'.pth')
            torch.save(D2.state_dict(), 'checkpoints/'+save_model_name+'_D2_'+str(epoch)+'.pth')
            G1.eval()
            G2.eval()
            evaluate(G1, G2, val_dataset, device, '{:s}/val_{:d}'.format('result', epoch))

    return G1, G2, D1, D2



def main(parser):
    G1 = Generator(input_channels=3, output_channels=1)
    G2 = Generator(input_channels=4, output_channels=3)
    D1 = Discriminator(input_channels=4)
    D2 = Discriminator(input_channels=7)

    '''load'''
    if parser.load is not None:
        print('load checkpoint ' + parser.load)

        G1_weights = torch.load('./checkpoints/ST-CGAN_G1_'+parser.load+'.pth')
        G1.load_state_dict(fix_model_state_dict(G1_weights))

        G2_weights = torch.load('./checkpoints/ST-CGAN_G2_'+parser.load+'.pth')
        G2.load_state_dict(fix_model_state_dict(G2_weights))

        D1_weights = torch.load('./checkpoints/ST-CGAN_D1_'+parser.load+'.pth')
        D1.load_state_dict(fix_model_state_dict(D1_weights))

        D2_weights = torch.load('./checkpoints/ST-CGAN_D2_'+parser.load+'.pth')
        D2.load_state_dict(fix_model_state_dict(D2_weights))

    train_img_list, val_img_list = make_datapath_list(phase='train', rate=parser.hold_out_ratio)

    mean = (0.5,)
    std = (0.5,)
    size = parser.image_size
    crop_size = parser.crop_size
    batch_size = parser.batch_size
    num_epochs = parser.epoch

    train_dataset = ImageDataset(img_list=train_img_list,
                                img_transform=ImageTransform(size=size, crop_size=crop_size, mean=mean, std=std),
                                phase='train')
    val_dataset = ImageDataset(img_list=val_img_list,
                                img_transform=ImageTransform(size=size, crop_size=crop_size, mean=mean, std=std),
                                phase='val')

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True) #num_workers=4

    G1_update, G2_update, D1_update, D2_update = train_model(G1, G2, D1, D2, dataloader=train_dataloader,
                                                            val_dataset=val_dataset, num_epochs=num_epochs,
                                                            parser=parser, save_model_name='ST-CGAN')

if __name__ == "__main__":
    parser = get_parser().parse_args()
    main(parser)
