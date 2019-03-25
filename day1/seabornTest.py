import inline as inline
import pandas as pd #数据分析
import seaborn as sns
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# def sinplot(flip=1):
#     x=np.linspace(0,14,100)
#     for i in range(1,7):
#         plt.plot(x,np.sin(x+i*.5)*(7-i)*flip)
#
# sns.set()
# plt.subplot2grid((2,3),(0,0))
# sinplot()
#
# plt.subplot2grid((2,3),(0,1))
# data = np.random.normal(size=(10,6))+np.arange(6)/2
# sns.boxplot(data=data)
# plt.show()
# print(np.arange(6))
# sns.set()
# sns.set_style("whitegrid")
# data = np.random.normal(size=(20,6))+np.arange(6)/2
# sns.boxplot(data=data)
data_train = pd.read_csv("Train.csv")
# AE = data_train[['Age','Fare']]#只要部分列的数据
# print(AE)
# sns.jointplot(x="Age",y="Fare",data=AE)
# plt.show()
#
# AE = data_train[['Age','Survived']]#只要部分列的数据
# print(AE)
# sns.jointplot(x="Age",y="Survived",data=AE,kind='hex')
# # sns.
# plt.show()
#
# # 多属性两两数据关系对比散点图
# ASFP = data_train[['Age','Survived',"Fare","Pclass"]]#只要部分列的数据
# sns.pairplot(ASFP)#四个属性之间两两关系图
# plt.show()

# 有一条线性回归的直线
AF = data_train[['Age',"Survived"]]#只要部分列的数据
sns.regplot(x="Age",y="Survived",data=AF,y_jitter=.2)#x_jitter,y_jitter指定在x轴或y轴上的浮动防止点位覆盖
plt.show()
