# -*- coding: utf-8 -*-
# 识别人脸 将输入图与库中的图进行对比找到相似度满足条件的输出
# 导入库
import os
import face_recognition

# 制作所有可用图像的列表
images = os.listdir('images')
# 加载图像
image_to_be_matched = face_recognition.load_image_file('my_image.jpg')

# 将加载图像编码为特征向量

image_to_be_matched_encoded = face_recognition.face_encodings(

    image_to_be_matched)[0]

# 遍历每张图像
for image in images:
    # 加载图像
    current_image = face_recognition.load_image_file("images/" + image)
    # 将加载图像编码为特征向量
    current_image_encoded = face_recognition.face_encodings(current_image)[0]

    # 将你的图像和图像对比，看是否为同一人

    result = face_recognition.compare_faces(

        [image_to_be_matched_encoded], current_image_encoded,0.4)

    # 检查是否一致

    if result[0] == True:

        print("Matched: " + image)

    else:

        print("Not matched: " + image)