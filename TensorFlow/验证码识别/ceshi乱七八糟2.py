#!/usr/bin/python
#  coding:utf-8
import tensorflow as tf
import numpy as np


def sparse_tuple_from(sequences, dtype=np.int32):
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), indices.max(0)[1] + 1], dtype=np.int64)
    # return tf.SparseTensor(indices=indices, values=values, shape=shape)
    return indices, values, shape
sq = [[0,1,2,3,4], [5,6,7,8,]]
indices, values, shape = sparse_tuple_from(sq)
print(indices)
print(values)
print(shape)
# images = np.random.random([6,2])*100
# # images2 = (images - np.mean(images)) / np.std(images)
# print("images:",images)
# print("images:",images[-2:])
# print("images:",images[:2])
# print("images2:",images2)
# sample_shape = np.asarray(enumerate(images)).shape[1:]
# print("images2shape:",(np.ones((8, 512) + sample_shape) * 0))
# label = np.asarray(range(0, 5))
# print([images, label])
# images = tf.cast(images, tf.float32)
# label = tf.cast(label, tf.int32)
# input_queue = tf.train.slice_input_producer([images, label], shuffle=False)
# # 将队列中数据打乱后再读取出来
# image_batch, label_batch = tf.train.shuffle_batch(input_queue, batch_size=10, num_threads=1, capacity=64, min_after_dequeue=1)
# with tf.Session() as sess:
#     # 线程的协调器
#     coord = tf.train.Coordinator()
#     # 开始在图表中收集队列运行器
#     threads = tf.train.start_queue_runners(sess, coord)
#     input_queue_v,image_batch_v, label_batch_v = sess.run([input_queue,image_batch, label_batch])
#     for j in range(5):
#         # print(image_batch_v.shape, label_batch_v[j])
#         print(image_batch_v[j]),
#         print(label_batch_v[j])
#     # 请求线程结束
#     coord.request_stop()
#     # 等待线程终止
#     coord.join(threads)