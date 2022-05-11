import os
import random
import numpy as np
import imageio
import scipy.misc as misc
from skimage.metrics import peak_signal_noise_ratio as compare_psnr # 后面导入
import skimage.measure as measure
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from carn.dataset import TrainDataset, TestDataset
from matplotlib import pyplot as plt
from PIL import Image
import torchvision.transforms as transforms


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
        train_loss = []
        epochs = 0
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        while epochs < 2:
            all_loss = 0
            kk = 0
            for inputs in self.train_loader:
                self.refiner.train()

                if cfg.scale > 0:
                    scale = cfg.scale
                    hr, lr = inputs[-1][0], inputs[-1][1]

                else:
                    # only use one of multi-scale data
                    # i know this is stupid but just temporary
                    scale = random.randint(2, 4)
                    # scale = 4
                    hr, lr = inputs[scale-2][0], inputs[scale-2][1]
                    # print(type(hr))  为tensor

                hr = hr.to(self.device)
                lr = lr.to(self.device)

                # .............
                # lr_ = lr[0]
                # lr_ = lr_.unsqueeze(0)
                # hr_ = hr[0]
                # hr_ = hr_.unsqueeze(0)
                # sr_ = refiner(lr_, scale)
                # loss_ = self.loss_fn_(sr_, hr_)
                # print('loss1: ', loss_)
                #
                # lr__ = lr[1]
                # lr__ = lr__.unsqueeze(0)
                # hr__ = hr[1]
                # hr__ = hr__.unsqueeze(0)
                # sr__ = refiner(lr__, scale)
                # loss__ = self.loss_fn__(sr__, hr__)
                # print('loss2: ', loss__)
                # ..........
                # lr chw
                sr = refiner(lr, scale)

                loss = self.loss_fn(sr, hr)

                # .....
                # sr_ = sr.cpu()
                # sr_ = sr_.mul(255).clamp(0, 255).byte()
                # sr_ = sr_.detach().numpy().transpose(0, 2, 3, 1)
                # im = Image.fromarray(np.uint8(sr_[0])).convert('RGB')
                # im.save('test_{}'.format(kk) + "_{}.png".format(epochs))
                # kk += 1
                # .....



                all_loss += loss.item()
                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.refiner.parameters(), cfg.clip)
                # 此处nn.utils.clip_grad_norm 被替换为nn.utils.clip_grad_norm_.
                self.optim.step()

                learning_rate = self.decay_learning_rate()  # lr decay
                for param_group in self.optim.param_groups:
                    param_group["lr"] = learning_rate  # 修改lr
                
                self.step += 1

                # ..
                dirpath = "E:/project/datasets/ASR data/ASR data_/train_LR_bicubic/X4/0000122_01200_d_0000119_LR.png"
                img_PIL = imageio.imread(dirpath)  # hwc
                img_PIL = transform(img_PIL).unsqueeze(0)  #  batchsize,c,h,w
                img_PIL = img_PIL.to(self.device)  # batch,c,h,w
                sr__ = refiner(img_PIL, scale)
                sr__= sr__.cpu()
                sr__ = sr__.mul(255).clamp(0, 255).byte()
                sr__ = sr__.detach().numpy().transpose(0, 2, 3, 1)
                im = Image.fromarray(np.uint8(sr__[0])).convert('RGB')
                im.save('testbigimage_{}'.format(kk) + "_{}.png".format(epochs))
                kk += 1
                # ..

            epochs += 1
                # verbose是决定是否输出日志的参数
            if cfg.verbose :  # True 原本为 cfg.verbose
                if cfg.scale > 0:
                    # 下面的Urban100 被修改为Set5
                    psnr = self.evaluate("E:/project/datasets/ASR data/Set5", scale=cfg.scale, num_step=self.step)
                    self.writer.add_scalar("Set5", psnr, epochs)
                    print("psnr:", psnr)
                else:
                    psnr = [self.evaluate("E:/project/datasets/ASR data/Set5", scale=i, num_step=self.step) for i in range(2, 5)]
                    self.writer.add_scalar("Set5_2x", psnr[0], epochs)
                    self.writer.add_scalar("Set5_3x", psnr[1], epochs)
                    self.writer.add_scalar("Set5_4x", psnr[2], epochs)
                    print("psnr:", psnr)

                self.save(cfg.ckpt_dir, cfg.ckpt_name, epochs)

            train_loss.append(all_loss)
            print(epochs, '/', '2000', ' SR loss: ', all_loss)


        plt.plot(train_loss)
        plt.title('train loss of SR')
        plt.savefig('SRtrainloss.png')

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
            h_half, w_half = int(h/2), int(w/2)
            h_chop, w_chop = h_half + cfg.shave, w_half + cfg.shave

            # split large image to 4 patch to avoid OOM error
            lr_patch = torch.FloatTensor(4, 3, h_chop, w_chop)
            lr_patch[0].copy_(lr[:, 0:h_chop, 0:w_chop])
            lr_patch[1].copy_(lr[:, 0:h_chop, w-w_chop:w])
            lr_patch[2].copy_(lr[:, h-h_chop:h, 0:w_chop])
            lr_patch[3].copy_(lr[:, h-h_chop:h, w-w_chop:w])
            lr_patch = lr_patch.to(self.device)
            
            # run refine process in here!
            sr = self.refiner(lr_patch, scale).data
            
            h, h_half, h_chop = h*scale, h_half*scale, h_chop*scale
            w, w_half, w_chop = w*scale, w_half*scale, w_chop*scale
            
            # merge splited patch images
            result = torch.FloatTensor(3, h, w).to(self.device)
            result[:, 0:h_half, 0:w_half].copy_(sr[0, :, 0:h_half, 0:w_half])
            result[:, 0:h_half, w_half:w].copy_(sr[1, :, 0:h_half, w_chop-w+w_half:w_chop])
            result[:, h_half:h, 0:w_half].copy_(sr[2, :, h_chop-h+h_half:h_chop, 0:w_half])
            result[:, h_half:h, w_half:w].copy_(sr[3, :, h_chop-h+h_half:h_chop, w_chop-w+w_half:w_chop])
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
        out = (im.astype(np.float64)-min_val) / (max_val-min_val)
        return out
        
    im1 = im2double(im1)
    im2 = im2double(im2)
    psnr = compare_psnr(im1, im2, data_range=1) # 源代码为measure.compare_psnr
    return psnr


