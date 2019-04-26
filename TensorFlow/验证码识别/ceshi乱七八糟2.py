#!/usr/bin/python
#  coding:utf-8
import tensorflow as tf
import numpy as np
from PIL import Image
image_data = Image.open("captcha\images/0107.jpg")
# 因为我们后边要用的Alexnet_v2网络需要输入数据为244*244所以图片要被非同比例拉伸（原来是160*60）
image_data = image_data.resize(( 224, 224))
# 模式“L”为灰色图像，它的每个像素用8个bit表示，0表示黑，255表示白，其他数字表示不同的灰度（0-255之间）
r, g, b = image_data.split()
r_arr = np.array(r).reshape((1,224,224))
r1 = image_data.convert('L').split()
image = np.array(r1[0]).reshape((1,224,224))
image = image.astype( np.float32) / 255.0
image = np.subtract(image, 0.5)
image = np.multiply(image, 2.0)
print(image)
# print(r_arr)