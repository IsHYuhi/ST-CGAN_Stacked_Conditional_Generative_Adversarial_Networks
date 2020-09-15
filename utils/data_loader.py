import os
import glob
import torch
import torch.utils.data as data
from . import ISTD_transforms
from PIL import Image
import random
from torchvision import transforms
import matplotlib.pyplot as plt


def make_datapath_list(phase="train", rate=0.8):
    """
    make filepath list for train, validation and test images
    """
    random.seed(44)

    rootpath = './dataset/' + phase + '/'
    files_name = os.listdir(rootpath + phase + '_A')

    if phase=='train':
        random.shuffle(files_name)
    elif phase=='test':
        files_name.sort()

    path_A = []
    path_B = []
    path_C = []

    for name in files_name:
        path_A.append(rootpath + phase + '_A/'+name)
        path_B.append(rootpath + phase + '_B/'+name)
        path_C.append(rootpath + phase + '_C/'+name)

    num = len(path_A)

    if phase=='train':
        path_A, path_A_val = path_A[:int(num*rate)], path_A[int(num*rate):]
        path_B, path_B_val = path_B[:int(num*rate)], path_B[int(num*rate):]
        path_C, path_C_val = path_C[:int(num*rate)], path_C[int(num*rate):]
        path_list = {'path_A': path_A, 'path_B': path_B, 'path_C': path_C}
        path_list_val = {'path_A': path_A_val, 'path_B': path_B_val, 'path_C': path_C_val}
        return path_list, path_list_val

    elif phase=='test':
        path_list = {'path_A': path_A, 'path_B': path_B, 'path_C': path_C}
        return path_list

class ImageTransformOwn():
    """
    preprocessing images for own images
    """
    def __init__(self, size=256, mean=(0.5, ), std=(0.5, )):
        self.data_transform = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize(mean, std)])

    def __call__(self, img):
        return self.data_transform(img)


class ImageTransform():
    """
    preprocessing images
    """
    def __init__(self, size=286, crop_size=256, mean=(0.5, ), std=(0.5, )):
        self.data_transform = {'train': ISTD_transforms.Compose([ISTD_transforms.Scale(size=size),
                                                            ISTD_transforms.RandomCrop(size=crop_size),
                                                            ISTD_transforms.RandomHorizontalFlip(p=0.5),
                                                            ISTD_transforms.ToTensor(),
                                                            ISTD_transforms.Normalize(mean, std)]),

                                'val': ISTD_transforms.Compose([ISTD_transforms.Scale(size=size),
                                                           ISTD_transforms.RandomCrop(size=crop_size),
                                                           ISTD_transforms.ToTensor(),
                                                           ISTD_transforms.Normalize(mean, std)]),

                                'test': ISTD_transforms.Compose([ISTD_transforms.Scale(size=size),
                                                            ISTD_transforms.RandomCrop(size=crop_size),
                                                            ISTD_transforms.ToTensor(),
                                                            ISTD_transforms.Normalize(mean, std)])}

    def __call__(self, phase, img):
        return self.data_transform[phase](img)


class ImageDataset(data.Dataset):
    """
    Dataset class. Inherit Dataset class from PyTrorch.
    """
    def __init__(self, img_list, img_transform, phase):
        self.img_list = img_list
        self.img_transform = img_transform
        self.phase = phase

    def __len__(self):
        return len(self.img_list['path_A'])

    def __getitem__(self, index):
        '''
        get tensor type preprocessed Image
        '''
        img = Image.open(self.img_list['path_A'][index]).convert('RGB')
        gt_shadow = Image.open(self.img_list['path_B'][index])
        gt = Image.open(self.img_list['path_C'][index]).convert('RGB')

        img, gt_shadow, gt = self.img_transform(self.phase, [img, gt_shadow, gt])

        return img, gt_shadow, gt

if __name__ == '__main__':
    img = Image.open('../dataset/train/train_A/test.png').convert('RGB')
    gt_shadow = Image.open('../dataset/train/train_B/test.png')
    gt = Image.open('../dataset/train/train_C/test.png').convert('RGB')

    print(img.size)
    print(gt_shadow.size)
    print(gt.size)

    f = plt.figure()
    f.add_subplot(1, 3, 1)
    plt.imshow(img)
    f.add_subplot(1, 3, 2)
    plt.imshow(gt_shadow, cmap='gray')
    f.add_subplot(1, 3, 3)
    plt.imshow(gt)

    img_transforms = ImageTransform(size=286, crop_size=256, mean=(0.5, ), std=(0.5, ))
    img, gt_shadow, gt = img_transforms([img, gt_shadow, gt])

    print(img.shape)
    print(gt_shadow.shape)
    print(gt.shape)


    f.add_subplot(2, 3, 4)
    plt.imshow(transforms.ToPILImage()(img).convert('RGB'))
    f.add_subplot(2, 3, 5)
    plt.imshow(transforms.ToPILImage()(gt_shadow).convert('L'), cmap='gray')
    f.add_subplot(2, 3, 6)
    plt.imshow(transforms.ToPILImage()(gt).convert('RGB'))
    f.tight_layout()
    plt.show()