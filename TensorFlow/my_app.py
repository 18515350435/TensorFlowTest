#!/usr/bin/python
#  coding:utf-8
from wsgiref.simple_server import make_server
import json
import urllib.parse as parse
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
def make_net():
    x=tf.placeholder(tf.float32,[None,784],name="x")
    y=tf.placeholder(tf.float32,[None,10],name="y")
    keep_prob=tf.placeholder(tf.float32,name="keep_prob")
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
    # 结果存放布尔列表中
    correct_predition =  tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
    # 准确率
    accuracy_ = tf.reduce_mean(tf.cast(correct_predition,tf.float32),name="accuracy")
    saver = tf.train.Saver()
    sess_ = tf.Session()
    saver.restore(sess_, "net/my_net.ckpt")
    return sess_,accuracy_
sess,accuracy = make_net()
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
def my_app(environ,start_response):
    # 载入数据集，如果没有则会将mnist数据下载到对应路径下
    acc = sess.run(sess.graph.get_tensor_by_name('accuracy:0'), feed_dict={'x:0': mnist.test.images, 'y:0': mnist.test.labels, 'keep_prob:0': 1.0})
    print(" acc:", acc)

    params = parse.parse_qs(environ['QUERY_STRING'])
    # 获取get中key为name的值
    name = params['name'][0]
    no = params['no'][0]

    # 组成一个数组，数组中只有一个字典
    dic = {'name': name, 'no': no," acc:":str(acc)}
    # 定义文件请求的类型和当前请求成功的code
    # start_response('200 OK', [('Content-Type', 'text/html; charset=utf-8')])
    start_response("200 OK", [('Content-Type','text/plain; charset=utf-8')])
    return [json.dumps(dic).encode()]
server = make_server("127.0.0.1", 8000, my_app)
server.serve_forever()