import pandas as pd
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
pd.set_option('display.max_columns',1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth',1000)
# 载入数据集，如果没有则会将mnist数据下载到对应路径下
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)
# 命名空间
with tf.name_scope('base_values'):
    #  批次大小
    batch_size = 100
    # 批次数
    n_batch = mnist.train.num_examples // batch_size
    keep_prob=tf.placeholder(tf.float32)
    lr = tf.Variable(0.0001,dtype=tf.float32)#步长

# 初始化权值
def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.1))
# 初始化偏置值
def bias_variable(shape):
    return tf.Variable(tf.constant(0.1,shape=shape))
# 卷积操作x和y方向步长都是1的SAME型的卷积操作
def conv2d(image,w):
    return tf.nn.conv2d(image,w,strides=[1,1,1,1],padding='SAME')
# 池化操作以最大值方式池化，窗口大小为2x2,x和y方向步长也都是2,'SAME'型的池化
def max_pool_2x2(image):
    return tf.nn.max_pool(image,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

# 命名空间
with tf.name_scope('input'):
    x=tf.placeholder(tf.float32,[None,784],name='x-input')
    y=tf.placeholder(tf.float32,[None,10],name='y-input')
# 将x转化为我们需要的4d向量
x_image = tf.reshape(x,[-1,28,28,1])
# ------------------第一个卷积层--------------------------
# 5x5x1 (1是第三个纬度此处可以指通道数，因为是灰度图所以通道数是1 ，如果三彩色的就是三基色通道数就是3) 的32个卷积核
w_conv1 = weight_variable([5,5,1,32])
# 每个卷积核一个偏置值
b_conv1 = bias_variable([32])
# 将x_image与权值向量进行卷积，再加上偏置值，然后用relu激活函数激活
conv2d_1 = conv2d(x_image,w_conv1)
h_conv1 = tf.nn.relu(conv2d_1+b_conv1)
h_pool1 = max_pool_2x2(h_conv1)#池化

# -----------------第二个卷积层---------------------------
# 5x5x32 (32是第三个纬度，上一层的卷积核扩增的纬度) 的64个卷积核
w_conv2 = weight_variable([5,5,32,64])
# 每个卷积核一个偏置值
b_conv2 = bias_variable([64])
# 将x_image与权值向量进行卷积，再加上偏置值，然后用relu激活函数激活
conv2d_2 = conv2d(h_pool1,w_conv2)
h_conv2 = tf.nn.relu(conv2d_2+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)#池化

# ---------------根据卷积结果进行神经网络计算----------------

with tf.name_scope('first_layer'):
    w1 = weight_variable([7*7*64,500])
    b1 = bias_variable([500])
    h_pool2_reshape = tf.reshape(h_pool2,[-1,7*7*64])
    L1 = tf.nn.relu(tf.matmul(h_pool2_reshape,w1)+b1)
    L1_drop = tf.nn.dropout(L1,keep_prob)
with tf.name_scope('output_layer'):
    w2 = weight_variable([500,10])
    b2 = bias_variable([10])
    prediction = tf.nn.softmax(tf.matmul(L1_drop,w2)+b2,name="prediction")
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
        # acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0})
        # print("epoch:",epoch," acc:",acc,"l_r:",l_r)