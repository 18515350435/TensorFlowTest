# 顺带一提引入tensorflow_hub时在模块搜索中要搜tensorflow-hub，代码中引入时要用import tensorflow_hub 其他类似
# github上下载TensorFlow 找到tensorflow\examples\image_retraining\retrain.py 可能目前转移到了hub中而且这个需要翻墙才能用
# 编写dos命令执行脚本：参数image_dir指定
# python C:\Users\admin\PycharmProjects\TensorFlowTestNew\hub-master\hub-master\examples\image_retraining\retrain.py --image_dir flower_photos
# 由于TensorFlow上最新的retrain.py要去寻找一个默认的在线的模型（tfhub_module的默认值），但是那个模型国内没有翻墙vpn无法访问到，
# 所以在网上找了个其他版本的retrain.py（https://github.com/googlecodelabs/tensorflow-for-poets-2/blob/master/scripts/retrain.py）能适用于TensorFlow_1.13.1，
# 并将里面的报错(报错原因是因为新旧版本方法位置变化了，当前使用的TensorFlow是1.13.1)予以修改,也可能不报错，之后得到了一个我自己的myretrain.py执行下边的脚本即可
# python C:\Users\admin\PycharmProjects\TensorFlowTestNew\TensorFlow\inception利用\myretrain.py --image_dir flower_photos --how_many_training_steps 200 --model_dir C:\Users\admin\PycharmProjects\TensorFlowTestNew\TensorFlow\inception_pretrain --output_graph output_graph.pd --output_labels output_labels.txt
# --image_dir flower_photos 要分类的图片地址
# --how_many_training_steps 200 训练周期是200，默认好像是4000
# --model_dir C:\Users\admin\PycharmProjects\TensorFlowTestNew\TensorFlow\inception_pretrain 我们要使用的模型的所在位置那个tgz包的位置，不指定回去自己下载一个
# --output_graph output_graph.pd 训练好的我自己的分类模型
# --output_labels output_labels.txt  输出一个标签位置
# 还有很多参数都有默认设置，到多数的输出文件位置都是默认在所在盘符的\tmp下，可以去代码中 argparse.ArgumentParser 下查看