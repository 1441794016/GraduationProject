import os
import glob
import h5py
import random
import numpy as np
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from yololoss import yolo_loss
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

        item = [random_crop(hr, lr, size, self.scale[i]) + (letterbox(lr_, 640, 32, auto=True)[0],) + (label,) + (label_size,) for
                i, (hr, lr, lr_, label, label_size) in enumerate(item)]
        item = [random_flip_and_rotate(hr, lr) + (lr_[:, :, ::-1].copy().transpose((2, 0, 1)),) + (label,) + (label_size,) for (hr, lr, lr_, label, label_size)
                in item]

        # 返回的是一个长度为3的list，list中的每个元素是长度为3的tuple，tuple中是hr,lr,lr_,label和label的行数，格式均为为tensor
        # lr变成了3*64*64大小，hr的大小根据scale可得到，lr和hr是经过随机翻转裁剪的，为chw格式；lr_为未经处理的图片且为hwc格式
        # label被填充为600*5的tensor
        return [(self.transform(hr), self.transform(lr), torch.from_numpy(lr_), torch.cat((label, torch.zeros(600 - int(label_size), 5)), 0), label_size) for (hr, lr, lr_, label, label_size) in item]

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


if __name__ == "__main__":
    device = 'cpu'
    train_data = TrainDataset("D:/SR/CARN-pytorch-master/dataset/Visdrone_train_.h5",
                              size=64,
                              scale=0)
    train_loader = DataLoader(train_data,
                              batch_size=4,
                              num_workers=0,
                              shuffle=True, drop_last=True)
    p = 0
    for inputs in train_loader:
        print(p)
        p = p + 1
        scale = random.randint(2, 4)
        hr, lr = inputs[scale - 2][0], inputs[scale - 2][1]
        lr_ = inputs[scale - 2][2]
        label = inputs[scale - 2][3]
        label_size = inputs[scale - 2][4]
        print(hr.size())
        print(label.size())

        i = 0
        new_label = torch.zeros(4, 600, 6)
        for data in label:
            image_index = [i] * 600
            image_index = torch.tensor(np.array(image_index))
            image_index = image_index.view(600, 1)
            new_label[i] = torch.cat((image_index, label[i]), 1)
            i = i + 1
        # lr_ = lr_.to(device, non_blocking=True).float() / 255
        print(new_label.size())
        print(lr.size())
        print(lr_.shape)
        print(type(lr_))
        #lr_ = lr_.numpy()
        print(type(lr_))
        print(label_size.size())


        for k in range(4):
            lr_img_in_batch = lr_[k]
            lr_img_in_batch = lr_img_in_batch.unsqueeze(0)
            lr_img_in_batch = lr_img_in_batch.numpy()
            label_in_batch = new_label[k]
            label_size_in_batch = int(label_size[k])
            label_in_batch = label_in_batch[:label_size_in_batch, :]
            print(label_in_batch.size())
            if k == 0:
                all_label_in_batch = label_in_batch
            if k != 0:
                all_label_in_batch = torch.cat((all_label_in_batch, label_in_batch), 0)

        print(all_label_in_batch.size())
        lr_ = lr_.numpy()
        loss = yolo_loss(lr_, all_label_in_batch, device)
        print('loss: ', loss)


