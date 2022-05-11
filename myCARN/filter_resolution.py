import os
from PIL import Image
import cv2
from shutil import copy


def filter_resolution(path, new_path):
    # path为原图片文件夹, newpath为筛选后的图片所在的文件夹
    for file in os.listdir(path):
        img = cv2.imread(path + '/' + file)
        h, w, c = img.shape
        print(w, ' ', h, " ", c)
        if w >= 1920 and h >= 1080:
            cv2.imwrite(new_path + '/' + file, img)
        print(file)

    print('筛选完成....')

def get_lable(path, path1, new_path):
    # 用于筛出图片的标签
    for file in os.listdir(path):
        file = os.path.splitext(file)[0]
        old = path1 + '/' + file + '.txt'
        new = new_path + '/' + file + '.txt'
        copy(old, new)
        print(file)
    print('标签筛选完成....')

def updata_label(path1, path2, path3):
    for file in os.listdir(path1):
        file = os.path.splitext(file)[0]
        old = path2 + '/' + file + '.txt'
        new = path3 + '/' + file + '.txt'
        copy(old, new)

if __name__=="__main__":
    path1 = "D:/SR/CARN-pytorch-master/dataset/ASR data_/train_HR_label"
    path2 = 'E:/datasets/VisDrone/VisDrone2019-DET-train/labels'
    path3 = "D:/SR/CARN-pytorch-master/dataset/ASR data_/train_HR_label_"
    updata_label(path1, path2, path3)