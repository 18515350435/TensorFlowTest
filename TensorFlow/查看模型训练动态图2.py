import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector
# 载入数据集，如果没有则会将mnist数据下载到对应路径下
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

# number of cycles
max_steps = 1001
# number of pictures
image_num = 3000
# file directory
DIR = "C:/Users/admin/PycharmProjects/TensorFlowTestNew/TensorFlow/"

# define session
sess = tf.Session()

# load pictures
embedding = tf.Variable(tf.stack(mnist.test.images[:image_num]), trainable=False, name='embedding')


# 命名空间
with tf.name_scope('base_values'):
    #  批次大小
    batch_size = 100
    # 批次数
    n_batch = mnist.train.num_examples // batch_size
    keep_prob=tf.placeholder(tf.float32)
    lr = tf.Variable(0.001,dtype=tf.float32)#步长

# 命名空间
with tf.name_scope('input'):
    x=tf.placeholder(tf.float32,[None,784],name='x-input')
    y=tf.placeholder(tf.float32,[None,10],name='y-input')

# show images 在tensorboard中显示
with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shaped_input, 10)

with tf.name_scope('first_layer'):
    # 使用tf.truncated_normal初始化很多时候会比使用tf.zeros好很多
    # tf.truncated_normal(shape, mean, stddev) :shape表示生成张量的维度，mean是均值，stddev是标准差。
    # 这个函数产生正太分布，均值和标准差自己设定。这是一个截断的产生正太分布的函数，就是说产生正太分布的值如果与均值的差值大于两倍的标准差，那就重新生成。
    # 和一般的正太分布的产生随机数据比起来，这个函数产生的随机数与均值的差距不会超过两倍的标准差，但是一般的别的函数是可能的。
    w1 = tf.Variable(tf.truncated_normal(shape=[784,500],stddev=0.1),name="w1")
    b1 = tf.Variable(tf.zeros([500])+0.1,name="b1")
    L1 = tf.nn.tanh(tf.matmul(x,w1)+b1,name="L1")
    L1_drop=tf.nn.dropout(L1,keep_prob)#相当于下一层的特征输入
with tf.name_scope('second_layer'):
    w2 = tf.Variable(tf.truncated_normal(shape=[500,300],stddev=0.1),name="w2")
    b2 = tf.Variable(tf.zeros([300])+0.1,name="b2")
    L2 = tf.nn.tanh(tf.matmul(L1_drop,w2)+b2,name="L2")
    L2_drop=tf.nn.dropout(L2,keep_prob,name="L2_drop")#相当于下一层的特征输入

with tf.name_scope('output_layer'):
    w3 = tf.Variable(tf.truncated_normal(shape=[300,10],stddev=0.1),name="w3")
    b3 = tf.Variable(tf.zeros([10])+0.1,name="b3")
    prediction = tf.nn.softmax(tf.matmul(L2_drop,w3)+b3,name="prediction")
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

# create metadata file
if tf.gfile.Exists(DIR + 'projector/projector/metadata.tsv'):
    tf.gfile.DeleteRecursively(DIR + 'projector/projector')
    tf.gfile.MkDir(DIR + 'projector/projector')
with open(DIR + 'projector/projector/metadata.tsv', 'w')  as f:
    labels = sess.run(tf.argmax(mnist.test.labels[:], 1))
    for i in range(image_num):
        f.write(str(labels[i]) + '\n')

# combine all summaries
merged = tf.summary.merge_all()

projector_writer = tf.summary.FileWriter(DIR + 'projector/projector', sess.graph)
saver = tf.train.Saver()
config = projector.ProjectorConfig()
embed = config.embeddings.add()
embed.tensor_name = embedding.name
embed.metadata_path = DIR + 'projector/projector/metadata.tsv'#测试集中对应的前image_num的label值
embed.sprite.image_path = DIR + 'projector/data/numbers.jpg'#这张图片顺序对应测试集的样本（测试集也是10000个样本）
# embed.sprite.image_path = DIR + 'projector/data/numberschild.jpg'
embed.sprite.single_image_dim.extend([28, 28])
projector.visualize_embeddings(projector_writer, config)

with tf.Session() as sess:
    sess.run(inti)
    tf.summary.FileWriter('logs/',sess.graph)#在terminal视图运行命令：tensorboard --logdir=C:\Users\admin\PycharmProjects\TensorFlowTestNew\TensorFlow\logs
    for epoch in range(1):
        sess.run(tf.assign(lr,lr*0.95))
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            sess.run([merged,train_step],feed_dict={x:batch_xs,y:batch_ys,keep_prob:1.0}, options=run_options,run_metadata=run_metadata)
        l_r = sess.run(lr);
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0})
        print("epoch:",epoch," acc:",acc,"l_r:",l_r)
    saver.save(sess, DIR + 'projector/projector/a_model.ckpt', global_step=max_steps)
    projector_writer.close