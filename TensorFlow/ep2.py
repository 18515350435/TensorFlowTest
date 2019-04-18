# coding: utf-8
import tensorflow as tf
import numpy as np
# x：2*4*3
x = tf.constant([[[1, 0, 1],
                  [1, 3, 1],
                  [0, 1, 1],
                  [1, 5, 1]],

                 [[1, 1, 1],
                  [1, 6, 1],
                  [1, 1, 1],
                  [1, 1, 1]]])
x1 = tf.reduce_sum(x)  # 6
x2 = tf.reduce_sum(x, 0)#去除第0维度 2*4*3 ==> 4*3
x3 = tf.reduce_sum(x, 1)#去除第1维度 2*4*3 ==> 2*3
x5 = tf.reduce_sum(x, 2) #去除第2维度 2*4*3 ==> 2*4
n = tf.not_equal(tf.cast(x, tf.float32), 0.)
w = tf.where(n)#返回为TRUE的位置坐标集合
g = tf.gather_nd(x, w)-1
target = tf.SparseTensor(indices=w, values=g, dense_shape=tf.cast(tf.shape(x), tf.int64))
# ................以此类推
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    # x1_ = sess.run(x1)
    # x2_ = sess.run(x2)
    # x3_ = sess.run(x3)
    # x5_ = sess.run(x5)
    n_ = sess.run(n)
    w_ = sess.run(w)
    g_ = sess.run(g)
    target_ = sess.run(target)
    # print(x1_)
    # print(x2_)
    # print(x3_)
    # print(x5_)
    # print(n_)
    # print(w_)
    print(target_)
