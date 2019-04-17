import tensorflow as tf
import numpy as np

p = tf.Variable(tf.random_normal([3, 1]))
b = tf.nn.embedding_lookup(p, [0])
c = tf.multiply(b*.1)
with tf.Session() as sess:
   sess.run(tf.global_variables_initializer())
   print(sess.run(p))
   print("-------------------------")

   print(sess.run(b))
   print("-------------------------")
   print(sess.run(p))