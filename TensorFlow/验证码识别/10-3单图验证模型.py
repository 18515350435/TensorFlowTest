# coding: UTF-8
import tensorflow as tf
from PIL import Image
from nets import nets_factory
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE = 1
CHAR_SET_LEN = 10
train_network_fn = nets_factory.get_network_fn(
    'alexnet_v2_captcha_multi',
    num_classes=CHAR_SET_LEN,
    weight_decay=0.0005,
    is_training=False)
# placeholder
x = tf.placeholder(tf.float32, [None, 224, 224])

# inputs: a tensor of size [batch_size, height, width, channels]
X = tf.reshape(x, [BATCH_SIZE, 224, 224, 1])
logits0, logits1, logits2, logits3, end_points = train_network_fn(X)

predict0 = tf.reshape(logits0, [-1, CHAR_SET_LEN])
predict0 = tf.argmax(predict0, 1)

predict1 = tf.reshape(logits1, [-1, CHAR_SET_LEN])
predict1 = tf.argmax(predict1, 1)

predict2 = tf.reshape(logits2, [-1, CHAR_SET_LEN])
predict2 = tf.argmax(predict2, 1)

predict3 = tf.reshape(logits3, [-1, CHAR_SET_LEN])
predict3 = tf.argmax(predict3, 1)

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, 'captcha/model/crack_captcha.model-1358')

    # coord = tf.train.Coordinator()
    # threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    image_data = Image.open("captcha\images/0107.jpg")
    # 展示图片
    plt.imshow(image_data)
    plt.axis('off')
    plt.show()
    # 因为我们后边要用的Alexnet_v2网络需要输入数据为244*244所以图片要被非同比例拉伸（原来是160*60）
    image_data = image_data.resize((224, 224))
    # 模式“L”为灰色图像，它的每个像素用8个bit表示，0表示黑，255表示白，其他数字表示不同的灰度（0-255之间）
    r1 = image_data.convert('L').split()
    image = np.array(r1[0]).reshape((1, 224, 224))
    image = image.astype(np.float32) / 255.0
    image = np.subtract(image, 0.5)
    imag_input = np.multiply(image, 2.0)
    label0, label1, label2, label3 = sess.run([predict0, predict1, predict2, predict3], feed_dict={x: imag_input})
    label0, label1, label2, label3 = sess.run([predict0, predict1, predict2, predict3], feed_dict={x: imag_input})
    print('predict:', label0, label1, label2, label3)
    # coord.request_stop()
    # coord.join(threads)