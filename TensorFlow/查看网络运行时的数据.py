import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# 载入数据集，如果没有则会将mnist数据下载到对应路径下
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)
# 命名空间
with tf.name_scope('base_values'):
    #  批次大小
    batch_size = 100
    # 批次数
    n_batch = mnist.train.num_examples // batch_size
    keep_prob=tf.placeholder(tf.float32)
    lr = tf.Variable(0.001,dtype=tf.float32)#步长
# 定义一个计算函数，用于分析数值变化
def variable_summaries(var,name):
    with tf.name_scope("summaries"+name):
        mean = tf.reduce_mean(var)#平均值
        with tf.name_scope("stddev"):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev',stddev)#标准差
        tf.summary.scalar('max',tf.reduce_max(var))#最大值
        tf.summary.scalar('min',tf.reduce_min(var))#最小值
        tf.summary.histogram('histogram',var)#直方图

# 命名空间
with tf.name_scope('input'):
    x=tf.placeholder(tf.float32,[None,784],name='x-input')
    y=tf.placeholder(tf.float32,[None,10],name='y-input')

with tf.name_scope('first_layer'):
    # 使用tf.truncated_normal初始化很多时候会比使用tf.zeros好很多
    # tf.truncated_normal(shape, mean, stddev) :shape表示生成张量的维度，mean是均值，stddev是标准差。
    # 这个函数产生正太分布，均值和标准差自己设定。这是一个截断的产生正太分布的函数，就是说产生正太分布的值如果与均值的差值大于两倍的标准差，那就重新生成。
    # 和一般的正太分布的产生随机数据比起来，这个函数产生的随机数与均值的差距不会超过两倍的标准差，但是一般的别的函数是可能的。
    w1 = tf.Variable(tf.truncated_normal(shape=[784,500],stddev=0.1),name="w1")
    b1 = tf.Variable(tf.zeros([500])+0.1,name="b1")
    L1 = tf.nn.tanh(tf.matmul(x,w1)+b1,name="L1")
    L1_drop=tf.nn.dropout(L1,keep_prob)#相当于下一层的特征输入

    variable_summaries(w1,"w1")
    variable_summaries(b1,"b1")
with tf.name_scope('second_layer'):
    w2 = tf.Variable(tf.truncated_normal(shape=[500,300],stddev=0.1),name="w2")
    b2 = tf.Variable(tf.zeros([300])+0.1,name="b2")
    L2 = tf.nn.tanh(tf.matmul(L1_drop,w2)+b2,name="L2")
    L2_drop=tf.nn.dropout(L2,keep_prob,name="L2_drop")#相当于下一层的特征输入

    variable_summaries(w2,"w2")
    variable_summaries(b2,"b2")

with tf.name_scope('output_layer'):
    w3 = tf.Variable(tf.truncated_normal(shape=[300,10],stddev=0.1),name="w3")
    b3 = tf.Variable(tf.zeros([10])+0.1,name="b3")
    prediction = tf.nn.softmax(tf.matmul(L2_drop,w3)+b3,name="prediction")

    variable_summaries(w3,"w3")
    variable_summaries(b3,"b3")
with tf.name_scope('train'):
    with tf.name_scope('loss'):
        # loss = tf.reduce_mean(tf.square(y-prediction))#二次代价函数
        loss =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))#对数似然代价函数（）或者说叫做用于softmax的交叉熵代价函数
    with tf.name_scope('train_step'):
        # train_step = tf.train.GradientDescentOptimizer(.1).minimize(loss)#梯度下降训练
        # train_step = tf.train.AdadeltaOptimizer(1e-3).minimize(loss)#Adadelta算法优化器训练
        train_step = tf.train.AdamOptimizer(lr).minimize(loss)#Adam算法优化器训练

    tf.summary.scalar('loss', loss)  # 损失值
with tf.name_scope('accuracy'):
    # 结果存放布尔列表中
    correct_predition =  tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
    # 准确率
    accuracy = tf.reduce_mean(tf.cast(correct_predition,tf.float32))
    tf.summary.scalar('accuracy', accuracy)  # 准确率

# 合并所有的summary
merged = tf.summary.merge_all()

with tf.name_scope('init_value'):
    inti = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(inti)
    writer = tf.summary.FileWriter('logs/',sess.graph)#在terminal视图运行命令：tensorboard --logdir=C:\Users\admin\PycharmProjects\TensorFlowTestNew\TensorFlow\logs
    for epoch in range(3):
        sess.run(tf.assign(lr,lr*0.95))
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            summary,_ = sess.run([merged,train_step],feed_dict={x:batch_xs,y:batch_ys,keep_prob:1.0})
            writer.add_summary(summary,epoch*n_batch+batch)
        l_r = sess.run(lr);
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0})
        print("epoch:",epoch," acc:",acc,"l_r:",l_r)