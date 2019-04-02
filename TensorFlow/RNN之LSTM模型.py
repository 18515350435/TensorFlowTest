import pandas as pd
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# LSTM 是长短期记忆网络，是一种时间递归神经网络，适合于处理和预测时间序列中间隔和延迟相对较长的重要事件
# 下边将一张图片的784个像素拆分成28次的顺序输入，每次输入28像素，来强行满足一把时间递归的操作，硬使用一次LSTM网络，来实现数字预测
pd.set_option('display.max_columns',1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth',1000)
# 载入数据集，如果没有则会将mnist数据下载到对应路径下
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)
n_inputs = 28 #每次输入的数据量
max_time = 28 #需要最大28次遍历一张图的所有数据
lstm_size = 100 #LSTM单元中的单元数。（可以简单理解为最终输出的信号特征100个）
n_classes = 10 #10个分类

# 命名空间
with tf.name_scope('base_values'):
    #  批次大小
    batch_size = 50
    # 批次数
    n_batch = mnist.train.num_examples // batch_size
    keep_prob=tf.placeholder(tf.float32)
    lr = tf.Variable(0.001,dtype=tf.float32)#步长

# 初始化权值
def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.1))
# 初始化偏置值
def bias_variable(shape):
    return tf.Variable(tf.constant(0.1,shape=shape))

# 命名空间
with tf.name_scope('input'):
    x=tf.placeholder(tf.float32,[None,784],name='x-input')
    y=tf.placeholder(tf.float32,[None,10],name='y-input')

# ---------------根据卷积结果进行神经网络计算----------------

with tf.name_scope('output_layer'):
    w1 = weight_variable([lstm_size,n_classes])
    b1 = bias_variable([n_classes])

inputs = tf.reshape(x,[-1,max_time,n_inputs])
# 定义lstm_size个LSTM基本CELL
lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)#1.13.1版本的调用方式
outputs,final_state =tf.nn.dynamic_rnn(lstm_cell, inputs,dtype=tf.float32)
prediction = tf.nn.softmax(tf.matmul(final_state[1],w1)+b1)

with tf.name_scope('train'):
    with tf.name_scope('loss'):
        # loss = tf.reduce_mean(tf.square(y-prediction))#二次代价函数
        loss =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))#对数似然代价函数（）或者说叫做用于softmax的交叉熵代价函数
    with tf.name_scope('train_step'):
        # train_step = tf.train.GradientDescentOptimizer(.1).minimize(loss)#梯度下降训练
        # train_step = tf.train.AdadeltaOptimizer(1e-3).minimize(loss)#Adadelta算法优化器训练
        train_step = tf.train.AdamOptimizer(lr).minimize(loss)#Adam算法优化器训练
with tf.name_scope('accuracy'):
    # 结果存放布尔列表中
    correct_predition =  tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
    # 准确率
    accuracy = tf.reduce_mean(tf.cast(correct_predition,tf.float32))
with tf.name_scope('init_value'):
    inti = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(inti)
    tf.summary.FileWriter('logs/',sess.graph)#在terminal视图运行命令：tensorboard --logdir=C:\Users\admin\PycharmProjects\TensorFlowTestNew\TensorFlow\logs
    for epoch in range(10):
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            # # 查看卷积池化值
            # conv2d_1_r, h_conv1_r, h_pool1_r = sess.run([conv2d_1,h_conv1,h_pool1],feed_dict={x:batch_xs,y:batch_ys,keep_prob:1.0})
            # conv2d_2_r, h_conv2_r, h_pool2_r = sess.run([conv2d_2,h_conv2,h_pool2],feed_dict={x:batch_xs,y:batch_ys,keep_prob:1.0})
            # f = open("out.txt", "w")
            # print("一层卷积结果：",conv2d_1_r,file=f)
            # print("一层激活结果：",h_conv1_r,file=f)
            # print("一层池化结果：",h_pool1_r,file=f)
            # print("二层卷积结果：",conv2d_2_r,file=f)
            # print("二层激活结果：",h_conv2_r,file=f)
            # print("二层池化结果：",h_pool2_r,file=f)

            # 将二次卷积池化后的结果放入神经网络训练
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,keep_prob:1.0})
        # l_r = sess.run(lr);
        finalState = sess.run([final_state],feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0})
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0})
        print("epoch:",epoch," acc:",acc,"lstmCell",lstm_cell,"finalState",finalState)
