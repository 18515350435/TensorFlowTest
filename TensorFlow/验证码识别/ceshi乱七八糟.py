# coding: UTF-8
from captcha.image import ImageCaptcha
import numpy as np
from PIL import Image
import random
import sys
import os
import tensorflow as tf
import matplotlib.pyplot as plt


# 定义一个解读函数
def read_and_decode(filename_quene, batch_size, shuffle_batch=True):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_quene)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image': tf.FixedLenFeature([], tf.string),
                                           'label0': tf.FixedLenFeature([], tf.int64),
                                           'label1': tf.FixedLenFeature([], tf.int64),
                                           'label2': tf.FixedLenFeature([], tf.int64),
                                           'label3': tf.FixedLenFeature([], tf.int64)
                                       })
    imageF = tf.decode_raw(features['image'], tf.int8)
    image = tf.reshape(imageF, [224, 224])
    label0 = tf.cast(features['label0'], tf.int32)
    label1 = tf.cast(features['label1'], tf.int32)
    label2 = tf.cast(features['label2'], tf.int32)
    label3 = tf.cast(features['label3'], tf.int32)

    # 随机抽取数据的代码应该写在函数外面，否则每抽取一次数据，就要载入一次tfrecord文件，会报异常
    #     if shuffle_batch:
    #         images, labels0, labels1, labels2, labels3 = tf.train.shuffle_batch([image, label0, label1, label2, label3],
    #                                                batch_size=batch_size,
    #                                                capacity=100,
    #                                                num_threads=4,
    #                                                min_after_dequeue=20)
    #     else:
    #         images, labels0, labels1, labels2, labels3 = tf.train.batch([image, label0, label1, label2, label3],
    #                                         batch_size=batch_size,
    #                                         capacity=100,
    #                                         num_threads=4,
    #                                         min_after_dequeue=20)
    return imageF,image, label0, label1, label2, label3


def show_tfrecord(tf_name, batch_num, batch_size):
    filename_queue = tf.train.string_input_producer([tf_name])
    imagesF,images, labels0, labels1, labels2, labels3 = read_and_decode(filename_queue, batch_size)
    images_b, labels0_b, labels1_b, labels2_b, labels3_b = tf.train.shuffle_batch(
        [images, labels0, labels1, labels2, labels3],
        batch_size=batch_size,
        capacity=100,
        num_threads=1,
        min_after_dequeue=20)
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(batch_num):
            # imagesF_, images_ = sess.run([imagesF,images])
            # plt.imshow(images_, cmap='gray')
            # plt.show()
            # imagesF_, images_ = sess.run([imagesF,images])
            # plt.imshow(images_, cmap='gray')
            # plt.show()
            img_, label0_, label1_, label2_, label3_ = sess.run([images_b, labels0_b, labels1_b, labels2_b, labels3_b])
            print('batch' + str(i) + ':')
            for j in range(batch_size):
                print(label0_[j], label1_[j], label2_[j], label3_[j])
                plt.imshow(img_[j], cmap='gray')
                plt.show()
        coord.request_stop()
        coord.join(threads)

show_tfrecord('captcha/train.tfrecord',batch_num=3,batch_size=3)
show_tfrecord('captcha/train.tfrecord',batch_num=3,batch_size=3)

# show_tfrecord('captcha/test.tfrecord', batch_num=2, batch_size=2)