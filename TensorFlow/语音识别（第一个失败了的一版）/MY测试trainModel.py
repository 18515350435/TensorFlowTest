# coding: utf-8
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm
import numpy as np
import os
from collections import Counter
import librosa  # https://github.com/librosa/librosa
# 数据下载地址 http://166.111.134.19:8081/data/thchs30/standalone.html
# 数据备份地址 Mega: https://mega.nz/#F!idRSjL4A!cnCY0R2NjU77Jr0soe9OgQ    Baidu: http://pan.baidu.com/s/1hqKwE00
# 训练样本路径
wav_path = 'data/wav/train2'
label_file = 'data/doc/trans/train.word.txt'

# 获得训练用的wav文件路径列表
def get_wav_files(wav_path=wav_path):
	wav_files = []
	for (dirpath, dirnames, filenames) in os.walk(wav_path):
		for filename in filenames:
			if filename.endswith('.wav') or filename.endswith('.WAV'):
				filename_path = os.sep.join([dirpath, filename])
				if os.stat(filename_path).st_size < 240000:  # 剔除掉一些小文件
					continue
				wav_files.append(filename_path)
	return wav_files

wav_files = get_wav_files()

# 读取wav文件对应的label
def get_wav_lable(wav_files=wav_files, label_file=label_file):
	labels_dict = {}
	with open(label_file, 'r', encoding='UTF-8') as f:
		for label in f:
			label = label.strip('\n')
			label_id = label.split(' ', 1)[0]
			label_text = label.split(' ', 1)[1]
			labels_dict[label_id] = label_text

	labels = []
	new_wav_files = []
	for wav_file in wav_files:
		wav_id = os.path.basename(wav_file).split('.')[0]
		if wav_id in labels_dict:
			labels.append(labels_dict[wav_id])
			new_wav_files.append(wav_file)

	return new_wav_files, labels
# wav_files音频文件路径集合
# labels   音频所对应的文字内容集合
wav_files, labels = get_wav_lable()
print("样本数:", len(wav_files))  # 8911
#print(wav_files[0], labels[0])
# wav/train/A11/A11_0.WAV -> 绿 是 阳春 烟 景 大块 文章 的 底色 四月 的 林 峦 更是 绿 得 鲜活 秀媚 诗意 盎然

# 词汇表(参看练习1和7)
all_words = []
for label in labels:
	all_words += [word for word in label]
counter = Counter(all_words) #分组计数 各个字出现的次数
count_pairs = sorted(counter.items(), key=lambda x: -x[1])

words, _ = zip(*count_pairs) #拿到字符词频由大到小的汉子集合
words_size = len(words)
print('词汇表大小:', words_size)
# 为汉子编号，频率越高，编号越靠前
word_num_map = dict(zip(words, range(len(words))))
to_num = lambda word: word_num_map.get(word, len(words))
# 将labels的文字变为文字编号形式的表示
labels_vector = [ list(map(to_num, label)) for label in labels]
#print(wavs_file[0], labels_vector[0])
#wav/train/A11/A11_0.WAV -> [479, 0, 7, 0, 138, 268, 0, 222, 0, 714, 0, 23, 261, 0, 28, 1191, 0, 1, 0, 442, 199, 0, 72, 38, 0, 1, 0, 463, 0, 1184, 0, 269, 7, 0, 479, 0, 70, 0, 816, 254, 0, 675, 1707, 0, 1255, 136, 0, 2020, 91]
#print(words[479]) #绿
label_max_len = np.max([len(label) for label in labels_vector])
print('最长句子的字数:', label_max_len)

wav_max_len = 673  # 673 可以将下边的计算wav_max_len的过程注释掉 直接赋值673
# wav_max_len = 0  # 673 可以将下边的计算wav_max_len的过程注释掉 直接赋值673
# for wav in wav_files:
#     wav, sr = librosa.load(wav, mono=True) #读取音频文件
#     mfcc = np.transpose(librosa.feature.mfcc(wav, sr), [1,0]) #将音频文件内容进行mfcc特征转化
#     if len(mfcc) > wav_max_len:
#         wav_max_len = len(mfcc)
print("最长的语音:", wav_max_len)

batch_size = 1
n_batch = len(wav_files) // batch_size

pointer = 0 #指针 指向下一个被从wav_files中读取的文件位置
# 获得一个batch
def get_next_batches(batch_size):
    global pointer
    batches_wavs = []
    batches_labels = []
    for i in range(batch_size):
        wav, sr = librosa.load(wav_files[pointer], mono=True)#读取音频文件
        mfcc = np.transpose(librosa.feature.mfcc(wav, sr), [1,0]) #将音频文件内容进行mfcc特征转化，transpose令定长的20作为最后一维的通道数
        batches_wavs.append(mfcc.tolist())
        batches_labels.append(labels_vector[pointer])
        pointer += 1

    # 补零对齐
    for mfcc in batches_wavs:
        while len(mfcc) < wav_max_len:
            mfcc.append([0]*20) # mfcc 默认的计算长度为20 作为channel length，所以每补充一行就20个0
    for label in batches_labels:
        while len(label) < label_max_len:
            label.append(0) #每段话不够也补充0,0代表无意义的空格（从前边的数据可以观察出来，因为空格最多所以编号是0）
    return batches_wavs, batches_labels

X = tf.placeholder(dtype=tf.float32, shape=[batch_size, wav_max_len, 20])#此处的None可以用wav_max_len代替
# sequence_len：mfcc特征属性的有效序列长度，那些补0的行不算在其中
sequence_len = tf.reduce_sum(tf.cast(tf.not_equal(tf.reduce_sum(X, reduction_indices=2), 0.), tf.int32), reduction_indices=1)
Y = tf.placeholder(dtype=tf.int32, shape=[batch_size, label_max_len])#此处的None可以用label_max_len代替

# conv1d_layer
conv1d_index = 0
def conv1d_layer(input_tensor, size, dim, activation, scale, bias):
    global conv1d_index
    with tf.variable_scope('conv1d_' + str(conv1d_index)):
        W = tf.get_variable('W', (size, input_tensor.get_shape().as_list()[-1], dim), dtype=tf.float32, initializer=tf.random_uniform_initializer(minval=-scale, maxval=scale))
        if bias:
            b = tf.get_variable('b', [dim], dtype=tf.float32, initializer=tf.constant_initializer(0))
        out = tf.nn.conv1d(input_tensor, W, stride=1, padding='SAME') + (b if bias else 0)
        if not bias:#如果不适用偏置值，就进行特征归一化操作
            # 特征归一化

            # beta = tf.get_variable('beta', dim, dtype=tf.float32, initializer=tf.constant_initializer(0))
            # gamma = tf.get_variable('gamma', dim, dtype=tf.float32, initializer=tf.constant_initializer(1))
            # mean_running = tf.get_variable('mean', dim, dtype=tf.float32, initializer=tf.constant_initializer(0))
            # variance_running = tf.get_variable('variance', dim, dtype=tf.float32, initializer=tf.constant_initializer(1))
            # mean, variance = tf.nn.moments(out, axes=list(range(len(out.get_shape()) - 1)))
            # def update_running_stat():
            #     decay = 0.99
            #     # 定义了均值方差指数衰减 见 http://blog.csdn.net/liyuan123zhouhui/article/details/70698264
            #     update_op = [mean_running.assign(mean_running * decay + mean * (1 - decay)), variance_running.assign(variance_running * decay + variance * (1 - decay))]
            #     # 指定先执行均值方差的更新运算 见 http://blog.csdn.net/u012436149/article/details/72084744
            #     with tf.control_dependencies(update_op):
            #         return tf.identity(mean), tf.identity(variance)
            # # 条件运算(https://applenob.github.io/tf_9.html) 这里指定为FALSE，所以一直是返回lambda: (mean_running, variance_running)，是不进行指数衰减的
            # m, v = tf.cond(tf.Variable(False, trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES]), update_running_stat, lambda: (mean_running, variance_running))
            # out = tf.nn.batch_normalization(out, m, v, beta, gamma, 1e-8)

            # *********上边注释掉的代码和下边的一行的作用一样（基本一样），特征归一化，特征值控制在[-1,1]之间********
            out = batch_norm(out, decay=0.99, updates_collections=None, is_training=True)
        if activation == 'tanh':
            out = tf.nn.tanh(out)
        if activation == 'sigmoid':
            out = tf.nn.sigmoid(out)

        conv1d_index += 1
        return out
# aconv1d_layer 空洞卷积 用一句话概括就是，在不用pooling的情况下扩大感受野（pooling层会导致信息损失）
# 在一般的卷积后我们习惯加上pooling层增加感受野，而空洞卷积则无需pooling层，rate参数是卷积核纬度扩增量，感受野因此过大了
# 参考 https://blog.csdn.net/guyuealian/article/details/86239099
aconv1d_index = 0
def aconv1d_layer(input_tensor, size, rate, activation, scale, bias):
    global aconv1d_index
    with tf.variable_scope('aconv1d_' + str(aconv1d_index)):
        shape = input_tensor.get_shape().as_list()
        W = tf.get_variable('W', (1, size, shape[-1], shape[-1]), dtype=tf.float32, initializer=tf.random_uniform_initializer(minval=-scale, maxval=scale))
        if bias:
            b = tf.get_variable('b', [shape[-1]], dtype=tf.float32, initializer=tf.constant_initializer(0))
        # tf.nn.atrous_conv2d空洞卷积操作，为了使用atrous_conv2d空洞卷积需要将纬度增加到2d+通道，所以扩增了一个维度
        out = tf.nn.atrous_conv2d(tf.expand_dims(input_tensor, dim=1), W, rate=rate, padding='SAME')
        out = tf.squeeze(out, [1])
        if not bias:
            # beta = tf.get_variable('beta', shape[-1], dtype=tf.float32, initializer=tf.constant_initializer(0))
            # gamma = tf.get_variable('gamma', shape[-1], dtype=tf.float32, initializer=tf.constant_initializer(1))
            # mean_running = tf.get_variable('mean', shape[-1], dtype=tf.float32, initializer=tf.constant_initializer(0))
            # variance_running = tf.get_variable('variance', shape[-1], dtype=tf.float32, initializer=tf.constant_initializer(1))
            # mean, variance = tf.nn.moments(out, axes=list(range(len(out.get_shape()) - 1)))
            # def update_running_stat():
            #     decay = 0.99
            #     update_op = [mean_running.assign(mean_running * decay + mean * (1 - decay)), variance_running.assign(variance_running * decay + variance * (1 - decay))]
            #     with tf.control_dependencies(update_op):
            #         return tf.identity(mean), tf.identity(variance)
            # m, v = tf.cond(tf.Variable(False, trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES]), update_running_stat, lambda: (mean_running, variance_running))
            # out = tf.nn.batch_normalization(out, m, v, beta, gamma, 1e-8)

            # *********上边注释掉的代码和下边的一行的作用一样（基本一样），特征归一化，特征值控制在[-1,1]之间********
            out = batch_norm(out, decay=0.99, updates_collections=None, is_training=True)
        if activation == 'tanh':
            out = tf.nn.tanh(out)
        if activation == 'sigmoid':
            out = tf.nn.sigmoid(out)

        aconv1d_index += 1
        return out
# 定义神经网络
def speech_to_text_network(n_dim=128, n_blocks=3):
    out = conv1d_layer(input_tensor=X, size=1, dim=n_dim, activation='tanh', scale=0.14, bias=False)
    # skip connections
    def residual_block(input_sensor, size, rate):
            conv_filter = aconv1d_layer(input_sensor, size=size, rate=rate, activation='tanh', scale=0.03, bias=False)
            conv_gate = aconv1d_layer(input_sensor, size=size, rate=rate,  activation='sigmoid', scale=0.03, bias=False)
            out = conv_filter * conv_gate
            out = conv1d_layer(out, size=1, dim=n_dim, activation='tanh', scale=0.08, bias=False)
            return out + input_sensor, out
    skip = 0
    for _ in range(n_blocks):
        for r in [1, 2, 4, 8, 16]:
            out, s = residual_block(out, size=7, rate=r)
            skip += s #这里不停地叠加是能更好的提取特征吗？如果上一行改成  _, s = residual_block(out, size=7, rate=r)呢？这样skip就变成入参input_sensor不变，而叠加不同空洞卷积的结果

    logit_ = conv1d_layer(skip, size=1, dim=skip.get_shape().as_list()[-1], activation='tanh', scale=0.08, bias=False)
    logit = conv1d_layer(logit_, size=1, dim=words_size, activation=None, scale=0.04, bias=True)

    return logit

class MaxPropOptimizer(tf.train.Optimizer):
    def __init__(self, learning_rate=0.001, beta2=0.999, use_locking=False, name="MaxProp"):
        super(MaxPropOptimizer, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta2 = beta2
        self._lr_t = None
        self._beta2_t = None
    def _prepare(self):
        self._lr_t = tf.convert_to_tensor(self._lr, name="learning_rate")
        self._beta2_t = tf.convert_to_tensor(self._beta2, name="beta2")
    def _create_slots(self, var_list):
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
    def _apply_dense(self, grad, var):
        lr_t = tf.cast(self._lr_t, var.dtype.base_dtype)
        beta2_t = tf.cast(self._beta2_t, var.dtype.base_dtype)
        if var.dtype.base_dtype == tf.float16:
            eps = 1e-7
        else:
            eps = 1e-8
        m = self.get_slot(var, "m")
        m_t = m.assign(tf.maximum(beta2_t * m + eps, tf.abs(grad)))
        g_t = grad / m_t
        var_update = tf.assign_sub(var, lr_t * g_t)
        return tf.group(*[var_update, m_t])
    def _apply_sparse(self, grad, var):
        return self._apply_dense(grad, var)

def train_speech_to_text_network():
    logit = speech_to_text_network()#通过卷积等操作提取特征

    # CTC时序分类算法 适合这种不知道输入输出是否对齐的情况(哪个字对应哪段声音)使用的算法，所以CTC适合语音识别和手写字符识别的任务
    indices = tf.where(tf.not_equal(tf.cast(Y, tf.float32), 0.))#因为0表示空格，此举是获取非空格的字符在Y中的位置坐标
    # tf.SparseTensor()定义一个稀疏张量indices:原张量中有效字符位置信息，values：原张量中有效字符集合(下边之所以减一是为了重新从0开始编号) dense_shape:原张量的形状
    target = tf.SparseTensor(indices=indices, values=tf.gather_nd(Y, indices) - 1, dense_shape=tf.cast(tf.shape(Y), tf.int64))
    # tf.nn.ctc_loss函数参考https://www.cnblogs.com/Libo-Master/p/8109691.html
    loss = tf.nn.ctc_loss(target,logit, sequence_len, time_major=False)
    loss = tf.reduce_mean(loss)
    # optimizer
    lr = tf.Variable(0.001, dtype=tf.float32, trainable=False)
    # optimizer = MaxPropOptimizer(learning_rate=lr, beta2=0.99)
    # var_list = [t for t in tf.trainable_variables()]
    # # optimizer.compute_gradients()梯度下降的开始一步，optimizer.apply_gradients()后续的梯度下降操作步骤 参考：https://blog.csdn.net/NockinOnHeavensDoor/article/details/80632677
    # gradient = optimizer.compute_gradients(loss, var_list=var_list)
    # optimizer_op = optimizer.apply_gradients(gradient)
    optimizer_op = tf.train.AdamOptimizer(lr).minimize(loss)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(tf.global_variables())

        for epoch in range(300):
            sess.run(tf.assign(lr, 0.001 * (0.97 ** epoch)))

            global pointer
            pointer = 0
            for batch in range(n_batch):
                batches_wavs, batches_labels = get_next_batches(batch_size)
                train_loss, _,logit_ = sess.run([loss, optimizer_op,logit], feed_dict={X: batches_wavs, Y: batches_labels})
                print(epoch, batch, train_loss)
            if epoch % 5 == 0:
                saver.save(sess, 'model/speech.model', global_step=epoch)

# 训练
# train_speech_to_text_network()

# 语音识别
# 把batch_size改为1
def speech_to_text(wav_file):
    wav, sr = librosa.load(wav_file, mono=True)  # 读取音频文件
    mfcc = np.transpose(librosa.feature.mfcc(wav, sr), [1, 0]).tolist()
    while len(mfcc) < wav_max_len:
        mfcc.append([0] * 20)
    mfcc = np.expand_dims(mfcc, axis=0)
    logit = speech_to_text_network() #网络最终的特征输出logit [batch_size,?,词汇表大小: 2666]

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, 'model/speech.model-0')

        decoded = tf.transpose(logit, perm=[1, 0, 2])
        shape = decoded[0].shape
        # 对输入中给出的logits执行波束搜索解码 tf.nn.ctc_beam_search_decoder函数参考：https://blog.csdn.net/qq_32791307/article/details/81037578
        decoded, _ = tf.nn.ctc_beam_search_decoder(inputs=decoded, sequence_length=sequence_len, merge_repeated=False)
        # predict = tf.sparse_to_dense(decoded[0].indices,shape, decoded[0].values) + 1
        output = sess.run(decoded, feed_dict={X: mfcc})
        for o in output[0].values:
            print(words[int(o+1)])

speech_to_text("data\wav/train\A2\A2_1.wav")
# 从麦克风获得语音输入，使用上面的模型进行识别。