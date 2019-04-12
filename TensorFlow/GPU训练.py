# coding: UTF-8

# 先看一下电脑是不是英伟达的显卡吧，目前只有英伟达nvidia的显卡支持CUDA和有效支持TensorFlow
# 安装参考资料 https://blog.csdn.net/dongyanwen6036/article/details/85056003

# 下载cuda（使用英伟达显卡加TensorFlow-gpu训练模型的所需环境）   地址： https://developer.nvidia.com/cuda-downloads 下载相关操作系统的版本
# 安装好之后吧CUDA安装目录下的bin和lib\x64添加进环境变量中

# 下载cudnn（加速训练速度）地址： https://developer.nvidia.com/rdp/cudnn-download 需注册登录后下载
# 将解压后文件夹中的bin、include、lib中的文件分别拷贝到C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1对应文件夹下
# 将C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\extras\CUPTI\lib64\cupti64_101.dll复制到C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\bin下
