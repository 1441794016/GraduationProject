import os

for file in os.listdir('E:/project/datasets/VisDrone/VisBicubic/x3/SR/images'):
    # print(file)
    st = file[0:23]
    old_name = 'E:/project/datasets/VisDrone/VisBicubic/x3/SR/images/' + file
    img_name = os.path.splitext(file)[0]  # 获取图片名称,不需要后缀
    new_name = 'E:/project/datasets/VisDrone/VisBicubic/x3/SR/images/' + st + '_SR.png'  # 新的图片名称的鹅拼接
    os.rename(old_name, new_name)  # 替换
print('重命名成功....')
