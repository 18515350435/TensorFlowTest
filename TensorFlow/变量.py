import tensorflow as tf

# 创建一个变量
x = tf.Variable([1,2]);
a = tf.constant([3,3])
sub=tf.subtract(x,a)
add = tf.add(x,sub)
init = tf.global_variables_initializer();
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(add))
