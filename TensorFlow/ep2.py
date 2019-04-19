# coding: utf-8
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm
def conv1d_layer():
   img = tf.Variable(tf.random_normal([ 2, 3])*10,name="img")

   beta = tf.get_variable('beta', 3, dtype=tf.float32, initializer=tf.constant_initializer(0))
   gamma = tf.get_variable('gamma', 3, dtype=tf.float32, initializer=tf.constant_initializer(1))
   mean_running = tf.get_variable('mean', 3, dtype=tf.float32, initializer=tf.constant_initializer(0))
   variance_running = tf.get_variable('variance', 3, dtype=tf.float32, initializer=tf.constant_initializer(1))
   mean, variance = tf.nn.moments(img, axes=list(range(len(img.get_shape()) - 1)))
   def update_running_stat():
       decay = 0.99
       # 定义了均值方差指数衰减 见 http://blog.csdn.net/liyuan123zhouhui/article/details/70698264
       update_op = [mean_running.assign(mean_running * decay + mean * (1 - decay)), variance_running.assign(variance_running * decay + variance * (1 - decay))]
       # 指定先执行均值方差的更新运算 见 http://blog.csdn.net/u012436149/article/details/72084744
       with tf.control_dependencies(update_op):
           return tf.identity(mean), tf.identity(variance)
   # 条件运算(https://applenob.github.io/tf_9.html) 这里指定为FALSE，所以一直是返回lambda: (mean_running, variance_running)，是不进行指数衰减的
   m, v = tf.cond(tf.Variable(True,name="flag",trainable=False), update_running_stat, lambda: (mean_running, variance_running))
   out1 = tf.nn.batch_normalization(img, m, v, beta, gamma, 1e-8)

   out2 =  batch_norm(img,decay=0.99,updates_collections=None,is_training=True)
   init = tf.global_variables_initializer()
   with tf.Session() as sess:
      sess.run(init)
      # out2_ = sess.run([v])
      img_,out1_, out2_ = sess.run([img,out1, out2])
      print("img_：", img_)
      print("out1_：", out1_)
      print("out2_：", out2_)
# axis = [0,1,2]#所以剩余的是四纬看做一个整体shape为[3]
# axis = [0,1]#所以剩余的是第三 四纬看做一个整体shape为[2, 3]
# mean, variance = tf.nn.moments(img, axis)
conv1d_layer()
