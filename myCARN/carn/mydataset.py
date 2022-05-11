import os
import glob
import h5py
import random
import numpy as np
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import torch
from utils.augmentations import letterbox

def random_crop(hr, lr, size, scale):
    h, w = lr.shape[:-1]
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)

    hsize = size * scale
    hx, hy = x * scale, y * scale

    crop_lr = lr[y:y + size, x:x + size].copy()
    crop_hr = hr[hy:hy + hsize, hx:hx + hsize].copy()

    return crop_hr, crop_lr


def random_flip_and_rotate(im1, im2):
    if random.random() < 0.5:
        im1 = np.flipud(im1)
        im2 = np.flipud(im2)

    if random.random() < 0.5:
        im1 = np.fliplr(im1)
        im2 = np.fliplr(im2)

    angle = random.choice([0, 1, 2, 3])
    im1 = np.rot90(im1, angle)
    im2 = np.rot90(im2, angle)

    # have to copy before be called by transform function
    return im1.copy(), im2.copy()


class TrainDataset(data.Dataset):
    def __init__(self, path, size, scale):
        super(TrainDataset, self).__init__()

        self.size = size
        h5f = h5py.File(path, "r")

        self.label = [v[:] for v in h5f["LABEL"].values()]
        # 将label由array变为tensor
        self.label = [torch.from_numpy(txt) for txt in self.label]
        self.hr = [v[:] for v in h5f["HR"].values()]
        # hr是一个大小为hr图片数量的list，每一个元素是array

        # 防止有的label只有一行导致读取的tensor变成一维，需要将他变成二维（1*5）
        ind = 0
        for data in self.label:
            if data.numel() == 5:
                self.label[ind] = self.label[ind].view(1, 5)
            ind = ind + 1

            # perform multi-scale training
        if scale == 0:
            self.scale = [2, 3, 4]
            self.lr = [[v[:] for v in h5f["X{}".format(i)].values()] for i in self.scale]
        else:
            self.scale = [scale]
            self.lr = [[v[:] for v in h5f["X{}".format(scale)].values()]]

        h5f.close()

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        size = self.size

        # 当scale = 0 时 ， i为0，1，2， 得到的item为一个长度为3的list（对应x2，x3，x4）
        # list中的元素是长度为3的tuple，tuple中是对应的hr,lr,lr,label和label的行数，label为tensor,其他为array
        item = [(self.hr[index], self.lr[i][index], self.lr[i][index], self.label[index],
                 torch.tensor(self.label[index].size(0))) for i, _ in enumerate(self.lr)]
        # scale = 0 时， item是一个长度为3的list，同上，是经过randomcrop后的图像

        item = [random_crop(hr, lr, size, self.scale[i]) + (lr_,) + (label,) + (label_size,) for
                i, (hr, lr, lr_, label, label_size) in enumerate(item)]
        item = [random_flip_and_rotate(hr, lr) + (lr_,) + (label,) + (label_size,) for (hr, lr, lr_, label, label_size)
                in item]

        # 返回的是一个长度为3的list，list中的每个元素是长度为3的tuple，tuple中是hr,lr,lr_,label和label的行数，格式均为为tensor
        # lr变成了3*64*64大小，hr的大小根据scale可得到，lr和hr是经过随机翻转裁剪的，为chw格式；lr_为未经处理的图片且为chw格式
        # label被填充为600*5的tensor
        return [(self.transform(hr), self.transform(lr), self.transform(lr_), torch.cat((label, torch.zeros(600 - int(label_size), 5)), 0), label_size) for (hr, lr, lr_, label, label_size) in item]

    def __len__(self):
        # 返回的大小就是数据集中训练图片的张数
        return len(self.hr)


class TestDataset(data.Dataset):
    def __init__(self, dirname, scale):
        super(TestDataset, self).__init__()

        self.name = dirname.split("/")[-1]
        self.scale = scale

        if "DIV" in self.name:
            self.hr = glob.glob(os.path.join("{}_HR".format(dirname), "*.png"))
            self.lr = glob.glob(os.path.join("{}_LR_bicubic".format(dirname),
                                             "X{}/*.png".format(scale)))
        else:
            all_files = glob.glob(os.path.join(dirname, "x{}/*.png".format(scale)))
            self.hr = [name for name in all_files if "HR" in name]
            self.lr = [name for name in all_files if "LR" in name]

        self.hr.sort()
        self.lr.sort()

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        hr = Image.open(self.hr[index])
        lr = Image.open(self.lr[index])

        hr = hr.convert("RGB")  # 转为RGB图像
        lr = lr.convert("RGB")
        filename = self.hr[index].split("/")[-1]

        return self.transform(hr), self.transform(lr), filename

    def __len__(self):
        return len(self.hr)

