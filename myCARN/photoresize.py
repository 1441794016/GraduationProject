from PIL import Image
import os.path
import glob

# 这个函数用于将不同尺寸的图片变为统一大小
def Resize(file, outdir, width, height):
    imgFile = Image.open(file)
    try:
        newImage = imgFile.resize((width, height), Image.BILINEAR)
        newImage.save(os.path.join(outdir, os.path.basename(file)))
    except Exception as e:
        print(e)

path = "D:\\SR\\CARN-pytorch-master\\dataset\\ASR data\\train_HR\\*.png"
for file in glob.glob(path):  # 图片所在的目录
    Resize(file, "D:\\SR\\CARN-pytorch-master\\dataset\\ASR data\\train_HR_", 1920, 1080)  # 新图片存放的目录,需要统一的尺寸