import tensorflow as tf
import numpy as np
# 100个样本训练集
x_data = np.random.rand(100)
y_data = x_data*0.1+0.2
# print(x_data)
# print(y_data)
# 构造模型
b = tf.Variable(0.)#定义截距变量b 初始化为0
k = tf.Variable(0.)#定义斜率变量k 初始化为0
y = k*x_data+b;
print(y)
# 二次代价函数
fc = tf.square(y_data-y)
loss = tf.reduce_mean(fc)
# 制定一个梯度下降的算法操作节点,(简单描述下梯度下降，就是对loss分别求k和b的偏导数,按照下边给定的0.2的步长不断迭代优化k和b的值)
optimizer = tf.train.GradientDescentOptimizer(0.2)
# print(optimizer)
# 最小化代价
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(400):#指定梯度下降次数400次
        if i%50==0:
            print(i,sess.run([k,b]))
        sess.run(train)
    print("最终的k和b的值", sess.run([k, b]))

