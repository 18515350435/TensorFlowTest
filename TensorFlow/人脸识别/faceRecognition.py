# ʶ������ ������ͼ����е�ͼ���жԱ��ҵ����ƶ��������������
# �����
import os
import face_recognition

# �������п���ͼ����б�
images = os.listdir('images')
# ����ͼ��
image_to_be_matched = face_recognition.load_image_file('my_image.jpg')

# ������ͼ�����Ϊ��������

image_to_be_matched_encoded = face_recognition.face_encodings(

    image_to_be_matched)[0]

# ����ÿ��ͼ��
for image in images:
    # ����ͼ��
    current_image = face_recognition.load_image_file("images/" + image)
    # ������ͼ�����Ϊ��������
    current_image_encoded = face_recognition.face_encodings(current_image)[0]

    # �����ͼ���ͼ��Աȣ����Ƿ�Ϊͬһ��

    result = face_recognition.compare_faces(

        [image_to_be_matched_encoded], current_image_encoded)

    # ����Ƿ�һ��

    if result[0] == True:

        print("Matched: " + image)

    else:

        print("Not matched: " + image)