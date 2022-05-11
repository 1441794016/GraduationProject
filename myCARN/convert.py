import cv2
import os
from PIL import Image


# 该函数用于将visdrone数据集的标签转化为yolo格式,对原visdrone数据集中score为0的不考虑
def convert(txt_path, img_path, w_path):
    f = open(txt_path, 'r')
    visdrone_data = f.readlines()
    new_data = []
    idx = 0
    for i in visdrone_data:
        new_data.append(i.split('\n')[0].split(','))
        idx = idx + 1
    f.close()

    img = Image.open(img_path)
    img_size = img.size
    img_w = img.width
    img_h = img.height

    yolotxt = []
    for data in new_data:
        if data[4] == '0':  # 注意不是data[0] == 0
            continue
        yolodata = []
        center_x = (float(data[0]) + float(data[2]) / 2) / img_w  # 归一化box中心横坐标
        center_y = (float(data[1]) + float(data[3]) / 2) / img_h  # 归一化box中心纵坐标
        box_w = float(data[2]) / img_w
        box_h = float(data[3]) / img_h
        id = data[5]
        yolodata = id + ' ' + str(center_x) + ' ' + str(center_y) + ' ' + str(box_w) + ' ' + str(box_h) + '\n'
        yolotxt.append(yolodata)

    yolotxt_file = open(w_path, 'w+')
    for i in range(len(yolotxt)):
        yolotxt_file.write(yolotxt[i])

    yolotxt_file.close()


# 该函数用于输出目录file_dir下的所有文件名称
def file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        # print(root) #当前目录路径
        # print(dirs) #当前路径下所有子目录
        # print(files)  # 当前路径下所有非目录子文件
        return files


# convert('D:/yolov3-master/data/annotations/0000006_00159_d_0000001.txt',
#        'D:/yolov3-master/data/images/0000006_00159_d_0000001.jpg',
#        'C:/Users/asus/Desktop/sample.txt')


txtdatafolder_path = 'D:/datasets/VisDrone/VisDrone2019-DET-test-dev/annotations'
imgdatafolder_path = 'D:/datasets/VisDrone/VisDrone2019-DET-test-dev/images'
newtxtdatafolder_path = 'D:/datasets/VisDrone/VisDrone2019-DET-test-dev/labels'
namelist = file_name(txtdatafolder_path)
l = len(namelist)
idx = 1
for i in namelist:
    re = i.split('.')[0] + '.jpg'
    convert(txtdatafolder_path + '/' + i, imgdatafolder_path + '/' + re, newtxtdatafolder_path + '/' + i)
    print(str(idx) + '/' + str(l))
    idx += 1
