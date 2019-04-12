# coding: UTF-8
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt

# 创建一个图来存放google调整好的模型 inception_pretrain\classify_image_graph_def.pb
# 结果数组与C:\Users\admin\PycharmProjects\TensorFlowTestNew\TensorFlow\inception利用\output_labels.txt文件中的顺序要一致
res = ['daisy','dandelion']
with tf.gfile.FastGFile('output_graph.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')#获取新模型最后的输出节点叫做final_result，可以从tensorboard中的graph中看到，其中名字后面的’:’之后接数字为EndPoints索引值（An operation allocates memory for its outputs, which are available on endpoints :0, :1, etc, and you can think of each of these endpoints as a Tensor.），通常情况下为0，因为大部分operation都只有一个输出。
    # 遍历目录
    for root, dirs, files in os.walk('testImage/'):#预测图片的位置
        for file in files:
            image_data = tf.gfile.FastGFile(os.path.join(root, file), 'rb').read()#Returns the contents of a file as a string.
            predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})#tensorboard中的graph中可以看到DecodeJpeg/contents是模型的输入变量名字
            predictions = np.squeeze(predictions)

            image_path = os.path.join(root, file)
            print(image_path)
            #展示图片
            # img = plt.imread(image_path)#只能读png图,所以不能显示其他图片，训练非png图时把这段注释掉，他只是一个显示作用
            # plt.imshow(img)
            # plt.axis('off')
            # plt.show()

            top_k = predictions.argsort()[-2:][::-1]#概率最高的后2个，然后在倒排一下
            for node_id in top_k:
                score = predictions[node_id]
                print('%s (score=%.5f)' % (res[node_id], score))
            print()
