# coding: UTF-8
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt

# 下载cuda（使用英伟达gpu训练模型的环境）   地址： https://developer.nvidia.com/cuda-downloads 下载相关操作系统的版本
# 安装好之后吧CUDA安装目录下的bin和lib\x64添加进环境变量中

# 下载cudnn（加速训练速度）地址： https://developer.nvidia.com/rdp/cudnn-download 需注册登录后下载