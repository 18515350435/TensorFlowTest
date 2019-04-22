# coding: utf-8
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm

W = tf.get_variable('W', (2, 3, 4), dtype=tf.int32,
                    initializer=tf.random_uniform_initializer(minval=-2, maxval=2))

indices = tf.where(tf.not_equal(tf.cast(W, tf.float32), 0.))  # 因为0表示空格，此举是获取非空格的字符在Y中的位置坐标
values=tf.gather_nd(W, indices)

with tf.Session() as sess:
   sess.run(tf.global_variables_initializer())
   print(W.eval())
   print(indices.eval())
   print(values.eval())