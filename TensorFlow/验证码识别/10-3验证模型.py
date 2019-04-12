# coding: UTF-8
import tensorflow as tf
from PIL import Image
from nets import nets_factory
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE = 1
TFRECORD_TEST_FILE = 'captcha/test.tfrecord'
CHAR_SET_LEN = 10

def read_and_decode(filename):
    # �����ļ�������һ������
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    # �����ļ������ļ�
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image' : tf.FixedLenFeature([], tf.string),
                                           'label0': tf.FixedLenFeature([], tf.int64),
                                           'label1': tf.FixedLenFeature([], tf.int64),
                                           'label2': tf.FixedLenFeature([], tf.int64),
                                           'label3': tf.FixedLenFeature([], tf.int64),
                                       })
    # ��ȡͼƬ����
    image = tf.decode_raw(features['image'], tf.uint8)
    # û�о���Ԥ����ĻҶ�ͼ
    image_raw = tf.reshape(image, [224, 224])
    # tf.train.shuffle_batch����ȷ��shape
    image = tf.reshape(image, [224, 224])
    # ͼƬԤ����
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    # ��ȡlabel
    label0 = tf.cast(features['label0'], tf.int32)
    label1 = tf.cast(features['label1'], tf.int32)
    label2 = tf.cast(features['label2'], tf.int32)
    label3 = tf.cast(features['label3'], tf.int32)

    return image, image_raw, label0, label1, label2, label3


# ��ȡͼƬ���ݺͱ�ǩ
image, image_raw, label0, label1, label2, label3 = read_and_decode(TFRECORD_TEST_FILE)

# ʹ��shuffle_batch�����������
image_batch, image_raw_batch, label_batch0, label_batch1, label_batch2, label_batch3 = tf.train.shuffle_batch(
    [image, image_raw, label0, label1, label2, label3], batch_size=BATCH_SIZE,
    capacity=1000, min_after_dequeue=200, num_threads=1)

# ��������ṹ
train_network_fn = nets_factory.get_network_fn(
    'alexnet_v2_captcha_multi',
    num_classes=CHAR_SET_LEN,
    weight_decay=0.0005,
    is_training=False)

# placeholder
x = tf.placeholder(tf.float32, [None, 224, 224])

# inputs: a tensor of size [batch_size, height, width, channels]
X = tf.reshape(x, [BATCH_SIZE, 224, 224, 1])
# ������������õ����ֵ
logits0, logits1, logits2, logits3, end_points = train_network_fn(X)

# Ԥ��ֵ
predict0 = tf.reshape(logits0, [-1, CHAR_SET_LEN])
predict0 = tf.argmax(predict0, 1)

predict1 = tf.reshape(logits1, [-1, CHAR_SET_LEN])
predict1 = tf.argmax(predict1, 1)

predict2 = tf.reshape(logits2, [-1, CHAR_SET_LEN])
predict2 = tf.argmax(predict2, 1)

predict3 = tf.reshape(logits3, [-1, CHAR_SET_LEN])
predict3 = tf.argmax(predict3, 1)

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, 'captcha/model/crack_captcha.model-1358')

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(10):
        # ��ȡһ�����ε����ݺͱ�ǩ
        b_image, b_image_raw, b_label0, b_label1, b_label2, b_label3 = sess.run([image_batch,
                                                                                 image_raw_batch,
                                                                                 label_batch0,
                                                                                 label_batch1,
                                                                                 label_batch2,
                                                                                 label_batch3])
        # ��ʾͼƬ
        img = Image.fromarray(b_image_raw[0], 'L')
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        # ��ӡ��ǩ
        print('label:', b_label0, b_label1, b_label2, b_label3)
        # Ԥ��
        label0, label1, label2, label3 = sess.run([predict0, predict1, predict2, predict3], feed_dict={x: b_image})
        # ��ӡԤ��ֵ
        print('predict:', label0, label1, label2, label3)
    coord.request_stop()
    coord.join(threads)