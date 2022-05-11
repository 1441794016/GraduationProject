from PIL import Image
import numpy as np
import math
import os

# 产生16个像素点不同的权重
def BiBubic(x):
    x = abs(x)
    if x <= 1:
        return 1 - 2 * (x ** 2) + (x ** 3)
    elif x < 2:
        return 4 - 8 * x + 5 * (x ** 2) - (x ** 3)
    else:
        return 0


# 双三次插值算法
# dstH为目标图像的高，dstW为目标图像的宽
def BiCubic_interpolation(img, dstH, dstW):
    scrH, scrW, _ = img.shape
    # img=np.pad(img,((1,3),(1,3),(0,0)),'constant')
    retimg = np.zeros((dstH, dstW, 3), dtype=np.uint8)
    for i in range(dstH):
        for j in range(dstW):
            scrx = i * (scrH / dstH)
            scry = j * (scrW / dstW)
            x = math.floor(scrx)
            y = math.floor(scry)
            u = scrx - x
            v = scry - y
            tmp = 0
            for ii in range(-1, 2):
                for jj in range(-1, 2):
                    if x + ii < 0 or y + jj < 0 or x + ii >= scrH or y + jj >= scrW:
                        continue
                    tmp += img[x + ii, y + jj] * BiBubic(ii - u) * BiBubic(jj - v)
            retimg[i, j] = np.clip(tmp, 0, 255)
    return retimg


if __name__ == "__main__":
    i = 0
    for file in os.listdir('E:/project/datasets/ASR data/Vis/x3'):
        # print(file)
        print(file)
        if "LR" in file:
            im_path = 'E:/project/datasets/ASR data/Vis/x3/' + file
            image = np.array(Image.open(im_path))
            image2 = BiCubic_interpolation(image, image.shape[0] * 3, image.shape[1] * 3)
            image2 = Image.fromarray(image2.astype('uint8')).convert('RGB')
            image2.save('E:/project/datasets/ASR data/test_visdrone x3/Bicubic Interpolation/Vis/x3/SR/' + file)
            i = i + 1
            print(i)
