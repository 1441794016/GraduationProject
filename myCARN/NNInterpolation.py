import cv2
import numpy as np

#dstH,dstW表示新图的Size，scrH,scrW表示旧图的Size
def NN_interpolation(img,dstH,dstW):
    scrH,scrW,_=img.shape
    retimg=np.zeros((dstH,dstW,3),dtype=np.uint8)
    for i in range(dstH-1):
        for j in range(dstW-1):
            #计算出新图坐标（i，j）坐标用旧图中的那个坐标来填充
            scrx=round(i*(scrH/dstH))
            scry=round(j*(scrW/dstW))
            retimg[i,j]=img[scrx,scry]
    return retimg


img = cv2.imread("C:/Users/asus/Desktop/zhanyong/0000221_00001_d_0000001_LR.png")
zoom = NN_interpolation(img,img.shape[0]*2,img.shape[1]*2)
cv2.imwrite("C:/Users/asus/Desktop/zhanyong/zoom.png", zoom)
cv2.imshow("nearest neighbor", zoom)
cv2.imshow("image", img)
cv2.waitKey(0)
