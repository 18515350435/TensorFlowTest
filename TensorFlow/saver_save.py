import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
# 载入数据集，如果没有则会将mnist数据下载到对应路径下
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)
#  批次大小
batch_size = 100
# 批次数
n_batch = mnist.train.num_examples // batch_size
x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,10])
keep_prob=tf.placeholder(tf.float32)

w1 = tf.Variable(tf.zeros([784,500]))
b1 = tf.Variable(tf.zeros([500])+0.1)
L1 = tf.nn.tanh(tf.matmul(x,w1)+b1)
L1_drop=tf.nn.dropout(L1,keep_prob)#相当于下一层的特征输入

w2 = tf.Variable(tf.zeros([500,300]))
b2 = tf.Variable(tf.zeros([300])+0.1)
L2 = tf.nn.tanh(tf.matmul(L1_drop,w2)+b2)
L2_drop=tf.nn.dropout(L2,keep_prob)#相当于下一层的特征输入

w3 = tf.Variable(tf.zeros([300,10]))
b3 = tf.Variable(tf.zeros([10])+0.1)

prediction = tf.nn.softmax(tf.matmul(L2_drop,w3)+b3)
# loss = tf.reduce_mean(tf.square(y-prediction))#二次代价函数
loss =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))#对数似然代价函数（）或者说叫做用于softmax的交叉熵代价函数

# train_step = tf.train.GradientDescentOptimizer(.1).minimize(loss)#梯度下降训练
# train_step = tf.train.AdadeltaOptimizer(1e-3).minimize(loss)#Adadelta算法优化器训练
train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)#Adam算法优化器训练
# 结果存放布尔列表中
correct_predition =  tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
# 准确率
accuracy = tf.reduce_mean(tf.cast(correct_predition,tf.float32))
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(20):
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,keep_prob:1.0})
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0})
        print("epoch:",epoch," acc:",acc)
    saver.save(sess,"net/my_net.ckpt")