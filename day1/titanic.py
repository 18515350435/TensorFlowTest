import pandas as pd #数据分析
import seaborn as sns
import numpy as np #科学计算
from pandas import Series,DataFrame
data_train = pd.read_csv("Train.csv")
print(data_train.Age.value_counts())
# print(type(data_train))
# data_train.info()
import matplotlib.pyplot as plt
fig = plt.figure()
fig.set(alpha=0.2) # 设定图表颜色alpha参数
plt.subplot2grid((3,3),(0,0), colspan=2) # 在一张大图里分列几个小图 colspan表示占几个小象限
# data_train.Survived.value_counts().plot(kind='bar')# 柱状图
data_train.Age.value_counts().plot(kind='kde')# 曲线图
plt.title(u"is Survived (1Survived)") # 标题
plt.ylabel(u"numbers")
plt.xlabel(u"age")


plt.subplot2grid((3,3),(1,0), colspan=2)
data_train.Age[data_train.Pclass == 1].plot(kind='kde')#指定多条子折线图Age后加了不同的筛选条件
data_train.Age[data_train.Pclass == 2].plot(kind='kde')
data_train.Age[data_train.Pclass == 3].plot(kind='kde')
plt.xlabel(u"age")# plots an axis lable
plt.ylabel(u"persent")
plt.title(u"per of age")
plt.legend((u'first', u'second',u'third'),loc='best') # sets our legend for our graph.

plt.subplot2grid((3,3),(0,2))
plt.scatter(data_train.Survived, data_train.Age,alpha=.1,norm=.3)
plt.ylabel(u"age") # 设定纵坐标名称
plt.grid(b=True, which='both', axis='y')
plt.title(u"age with Survived (1Survived)")

plt.subplot2grid((3,3),(1,2))
data_train.Embarked.value_counts().plot(kind='bar')
plt.title(u"Embarked")
plt.ylabel(u"number")

plt.show()

# plt.subplot2grid((3,3),(2,0),colspan=2)
AE = data_train[['Age','Fare']]#只要部分列的数据
print(AE)
sns.jointplot(x="Age",y="Fare",data=AE)
plt.show()

