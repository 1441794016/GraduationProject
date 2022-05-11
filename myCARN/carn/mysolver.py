import os
import random
import numpy as np
import time
import scipy.misc as misc
from skimage.metrics import peak_signal_noise_ratio as compare_psnr  # 后面导入
import skimage.measure as measure
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from mydataset import TrainDataset, TestDataset
from yololoss import yolo_loss
from matplotlib import pyplot as plt
from PIL import Image

class Solver():
    def __init__(self, model, cfg):
        if cfg.scale > 0:
            self.refiner = model(scale=cfg.scale,
                                 group=cfg.group)
        else:
            self.refiner = model(multi_scale=True,
                                 group=cfg.group)

        if cfg.loss_fn in ["MSE"]:
            self.loss_fn = nn.MSELoss()
        elif cfg.loss_fn in ["L1"]:
            self.loss_fn = nn.L1Loss()
        elif cfg.loss_fn in ["SmoothL1"]:
            self.loss_fn = nn.SmoothL1Loss()

        self.optim = optim.Adam(
            filter(lambda p: p.requires_grad, self.refiner.parameters()),
            cfg.lr)
        # 优化器使用adam
        self.train_data = TrainDataset(cfg.train_data_path,
                                       scale=cfg.scale,
                                       size=cfg.patch_size)
        self.train_loader = DataLoader(self.train_data,
                                       batch_size=cfg.batch_size,
                                       num_workers=1,
                                       shuffle=True, drop_last=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.refiner = self.refiner.to(self.device)
        self.loss_fn = self.loss_fn

        self.cfg = cfg
        self.step = 0

        self.writer = SummaryWriter(log_dir=os.path.join("runs", cfg.ckpt_name))
        if cfg.verbose:
            num_params = 0
            for param in self.refiner.parameters():
                num_params += param.nelement()
            print("# of params:", num_params)

        os.makedirs(cfg.ckpt_dir, exist_ok=True)

    def fit(self):
        cfg = self.cfg
        refiner = nn.DataParallel(self.refiner,
                                  device_ids=range(cfg.num_gpu))

        learning_rate = cfg.lr
        all_asr_train_loss = []
        all_sr_train_loss = []
        all_yolo_train_loss = []
        epochs = 0
        while epochs < 100:
            tic = time.time()
            asr_train_loss = 0
            sr_train_loss = 0
            yolo_train_loss = 0
            kk = 0
            for inputs in self.train_loader:
                self.refiner.train()

                if cfg.scale > 0:
                    scale = cfg.scale
                    hr, lr = inputs[-1][0], inputs[-1][1]
                    lr_ = inputs[-1][2]
                    label = inputs[-1][3]
                    label_size = inputs[-1][4]
                else:
                    # only use one of multi-scale data
                    # i know this is stupid but just temporary
                    scale = random.randint(2, 4)

                    hr, lr = inputs[scale - 2][0], inputs[scale - 2][1]
                    lr_ = inputs[scale - 2][2]
                    label = inputs[scale - 2][3]
                    label_size = inputs[scale - 2][4]

                i = 0
                new_label = torch.zeros(cfg.batch_size, 600, 6)
                for data in label:
                    image_index = [i] * 600
                    image_index = torch.tensor(np.array(image_index))
                    image_index = image_index.view(600, 1)
                    new_label[i] = torch.cat((image_index, label[i]), 1)
                    i = i + 1

                # lr_ = lr_.to(self.device, non_blocking=True).float() / 255

                for k in range(cfg.batch_size):
                    label_in_batch = new_label[k]
                    label_in_batch = label_in_batch[:int(label_size[k]), :]
                    if k == 0:
                        all_label_in_batch = label_in_batch
                    if k != 0:
                        all_label_in_batch = torch.cat((all_label_in_batch, label_in_batch), 0)

                # print(lr_.shape)
                lr_ = lr_.to(self.device).float()
                # print(lr_.shape) bchw
                # print(type(lr_))  torch.tensor
                sr_ = refiner(lr_, scale)
                sr_ = sr_.cpu().mul(255).clamp(0, 255).byte()
                sr_ = sr_.detach().numpy().transpose(0, 2, 3, 1)
                # 为(batchsize,h,w,c)

                im = Image.fromarray(np.uint8(sr_[0])).convert('RGB')
                if kk ==300:
                    im.save('test_{}'.format(kk) + "_{}.png".format(epochs))
                kk += 1

                # hr，lr为chw格式
                hr = hr.to(self.device)
                lr = lr.to(self.device)
                # 注意lr_为hwc格式

                sr = refiner(lr, scale)
                loss_sr = self.loss_fn(sr, hr)
                sr_train_loss = sr_train_loss + loss_sr.item()

                if epochs >= 0:
                    # yolo_loss 的输入是batchsize,h,w,c
                    loss_yolo = yolo_loss(sr_, all_label_in_batch, self.device)
                    yolo_train_loss = yolo_train_loss + loss_yolo.item()

                    loss = 0.9 * loss_sr + 0.1 * loss_yolo
                else:
                    loss = loss_sr


                asr_train_loss = asr_train_loss + loss.item()
                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.refiner.parameters(), cfg.clip)
                # 此处nn.utils.clip_grad_norm 被替换为nn.utils.clip_grad_norm_.
                self.optim.step()

                learning_rate = self.decay_learning_rate()  # lr decay
                for param_group in self.optim.param_groups:
                    param_group["lr"] = learning_rate  # 修改lr

                self.step += 1
                # verbose是决定是否输出日志的参数
            toc = time.time()
            epochs += 1
            if cfg.verbose:  # True 原本为 cfg.verbose
                if cfg.scale > 0:
                    # 下面的Urban100 被修改为Set5
                    psnr = self.evaluate("/home/mist/Set5", scale=cfg.scale,
                                         num_step=self.step)
                    self.writer.add_scalar("Set5", psnr, epochs)
                    print("psnr:", psnr)
                else:
                    psnr = [self.evaluate("/home/mist/Set5", scale=i, num_step=self.step) for
                            i in range(2, 5)]
                    self.writer.add_scalar("Set5_2x", psnr[0], epochs)
                    self.writer.add_scalar("Set5_3x", psnr[1], epochs)
                    self.writer.add_scalar("Set5_4x", psnr[2], epochs)
                    print("psnr:", psnr)

                self.save(cfg.ckpt_dir, cfg.ckpt_name, epochs)

            print(epochs, '/', 100, 'asr loss: ', asr_train_loss, ' ',
                  'sr loss: ', sr_train_loss, ' ', 'yolo loss: ', yolo_train_loss)
            print("time cost: ", toc-tic)
            all_asr_train_loss.append(asr_train_loss)
            all_sr_train_loss.append(sr_train_loss)
            all_yolo_train_loss.append(yolo_train_loss)

        plt.plot(all_asr_train_loss)
        #  plt.plot(all_sr_train_loss)
        #  plt.plot(all_yolo_train_loss)
        plt.title('ASR loss ')
        plt.legend(['ASR'])
        plt.savefig('ASRtrainloss.png')
        a = np.array(all_asr_train_loss)
        np.save('a.npy', a)

    def evaluate(self, test_data_dir, scale=2, num_step=0):
        cfg = self.cfg
        mean_psnr = 0
        self.refiner.eval()

        test_data = TestDataset(test_data_dir, scale=scale)
        test_loader = DataLoader(test_data,
                                 batch_size=1,
                                 num_workers=1,
                                 shuffle=False)

        for step, inputs in enumerate(test_loader):
            hr = inputs[0].squeeze(0)
            lr = inputs[1].squeeze(0)
            name = inputs[2][0]

            h, w = lr.size()[1:]
            h_half, w_half = int(h / 2), int(w / 2)
            h_chop, w_chop = h_half + cfg.shave, w_half + cfg.shave

            # split large image to 4 patch to avoid OOM error
            lr_patch = torch.FloatTensor(4, 3, h_chop, w_chop)
            lr_patch[0].copy_(lr[:, 0:h_chop, 0:w_chop])
            lr_patch[1].copy_(lr[:, 0:h_chop, w - w_chop:w])
            lr_patch[2].copy_(lr[:, h - h_chop:h, 0:w_chop])
            lr_patch[3].copy_(lr[:, h - h_chop:h, w - w_chop:w])
            lr_patch = lr_patch.to(self.device)

            # run refine process in here!
            sr = self.refiner(lr_patch, scale).data

            h, h_half, h_chop = h * scale, h_half * scale, h_chop * scale
            w, w_half, w_chop = w * scale, w_half * scale, w_chop * scale

            # merge splited patch images
            result = torch.FloatTensor(3, h, w).to(self.device)
            result[:, 0:h_half, 0:w_half].copy_(sr[0, :, 0:h_half, 0:w_half])
            result[:, 0:h_half, w_half:w].copy_(sr[1, :, 0:h_half, w_chop - w + w_half:w_chop])
            result[:, h_half:h, 0:w_half].copy_(sr[2, :, h_chop - h + h_half:h_chop, 0:w_half])
            result[:, h_half:h, w_half:w].copy_(sr[3, :, h_chop - h + h_half:h_chop, w_chop - w + w_half:w_chop])
            sr = result

            hr = hr.cpu().mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
            sr = sr.cpu().mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()

            # evaluate PSNR
            # this evaluation is different to MATLAB version
            # we evaluate PSNR in RGB channel not Y in YCbCR
            bnd = scale
            im1 = hr[bnd:-bnd, bnd:-bnd]
            im2 = sr[bnd:-bnd, bnd:-bnd]
            mean_psnr += psnr(im1, im2) / len(test_data)

        return mean_psnr

    def load(self, path):
        self.refiner.load_state_dict(torch.load(path))
        splited = path.split(".")[0].split("_")[-1]
        try:
            self.step = int(path.split(".")[0].split("_")[-1])
        except ValueError:
            self.step = 0
        print("Load pretrained {} model".format(path))

    def save(self, ckpt_dir, ckpt_name, epoch):
        save_path = os.path.join(
            ckpt_dir, "{}_{}.pth".format(ckpt_name, epoch))
        torch.save(self.refiner.state_dict(), save_path)

    def decay_learning_rate(self):
        lr = self.cfg.lr * (0.5 ** (self.step // self.cfg.decay))
        return lr


def psnr(im1, im2):
    def im2double(im):
        min_val, max_val = 0, 255
        out = (im.astype(np.float64) - min_val) / (max_val - min_val)
        return out

    im1 = im2double(im1)
    im2 = im2double(im2)
    psnr = compare_psnr(im1, im2, data_range=1)  # 源代码为measure.compare_psnr
    return psnr
