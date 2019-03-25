import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# a=[[-1.01,-2.01,-3.01],[3.01,4.01,5.01]]
# b=[[1.01,2.01,2.01,2.01],[1.01,3.01,3.01,2.01],[1.01,2.01,3.01,3.01]]
# ab=tf.matmul(a,b)
# ab2=tf.matmul(a,b)+[[.01,.01,.01,.01]]
# L1 = tf.nn.tanh(ab2);
# with tf.Session() as sess:
#     print(sess.run(ab))
#     print(sess.run(ab2))
#     print(sess.run(L1))

# 创建训练集
x_data = np.linspace(-1,1,200)[:,np.newaxis]
noise = np.random.normal(0,0.02,x_data.shape)
y_data = np.square(x_data)+noise

# 强行使用一波神经网络模型搞一波-----------------

x=tf.placeholder(tf.float32,[None,1])
y=tf.placeholder(tf.float32,[None,1])
# 神经网络中间层
Weights_L1 = tf.Variable(tf.random_normal([1,10]))
biases_L1 = tf.Variable(tf.zeros([1,10]))
Wx_plus_b_L1 = tf.matmul(x,Weights_L1)+biases_L1
L1 = tf.nn.tanh(Wx_plus_b_L1)#双曲正切函数,可用作激活函数
# 神经网络输出层
Weights_L2 = tf.Variable(tf.random_normal([10,1]))
biases_L2 = tf.Variable(tf.zeros([1,1]))
Wx_plus_b_L2 = tf.matmul(L1,Weights_L2)+biases_L2
prediction = tf.nn.tanh(Wx_plus_b_L2)#这里不适用双曲正切激活函数效果更好但是就不太算神经网络模型了
# 二次代价函数
loss = tf.reduce_mean(tf.square(y-prediction))
# 使用梯度下降训练
train_step = tf.train.GradientDescentOptimizer(.1).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(2000):
        sess.run(train_step,feed_dict={x:x_data,y:y_data})
    # 梯度下降2000次后使用当前的模型获取训练集的预测值
    prediction_v = sess.run(prediction,feed_dict={x:x_data})
    # 绘图对比真实值和预测值
    plt.figure()
    plt.scatter(x_data,y_data)
    plt.plot(x_data,prediction_v,'r-')
    plt.show()