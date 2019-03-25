import inline as inline
import pandas as pd #数据分析
import seaborn as sns
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


data_train = pd.read_csv("Train.csv")

# #兄弟姐妹数量与获救情况分析（兄弟姐妹越少获救几率稍微偏高，不一定最符合正相关）
# AF = data_train[['SibSp',"Survived"]]#只要部分列的数据
# sns.regplot(x="SibSp",y="Survived",data=AF,y_jitter=0.1,x_jitter=0.3)#x_jitter,y_jitter指定在x轴或y轴上的浮动防止点位覆盖
# plt.show()

# #年龄与获救情况分析（年龄越小获救几率越高，不一定最符合正相关）
# AF = data_train[['Age',"Survived"]]#只要部分列的数据
# sns.regplot(x="Age",y="Survived",data=AF,y_jitter=0.1,x_jitter=0.3)#x_jitter,y_jitter指定在x轴或y轴上的浮动防止点位覆盖
# plt.show()

#性别与获救情况分析（年龄越小获救几率越高，不一定最符合正相关）
# AF = data_train[['Sex',"Survived"]]#只要部分列的数据
# print(AF['Sex'].map(lambda x: x=='male'?'0':'1'))
# AF['Sex'].e
# AF['Sex']=(AF['Sex']=='female').astype('int')#将性别用01表示方便绘制离散图
# 当有好多种类是非数字表示时候，为了用regplot加x_jitter,y_jitter来看其分布
# data_train['SexNum'] = data_train['Sex'].replace(['male','female'],[0,1])
# print(AF)
# print(type(AF))
# print(AF.dtypes)
# print(AF)
# sns.regplot(x="Sex",y="Survived",data=AF,x_jitter=.2,y_jitter=.2)#x_jitter,y_jitter指定在x轴或y轴上的浮动防止点位覆盖
# plt.show()

#分类图，x,y至少有一个数值（个人感觉此图效果最佳），女性的获救情况比男性高很多
# sns.swarmplot(x='Sex',y="Age",data=data_train,hue="Survived")
# plt.show()
# sns.stripplot(x='Sex',y="Age",data=data_train,hue="Survived")
# plt.show()

# 小提琴图split=True会根据hue指定的属性分隔在两边
# sns.violinplot(x='Sex',y="Age",data=data_train,hue="Survived",split=True)
# plt.show()

# # 通过观察船舱号码的有无发现有船舱号的获救概率大
# data_train['IsCabin']=data_train['Cabin'].isnull()
# sns.regplot(x='IsCabin',y="Survived",data=data_train,y_jitter=.2,x_jitter=.2)
# plt.show()

# 不同的仓位等级以及性别对获救情况影响 因为Survived字段是0-1(1是获救) sns.barplot图是按照y属性求均值，在此处正好是获救占比
# sns.barplot(x='Sex',y='Survived',hue='Pclass',data=data_train)
# plt.show()

# 不同的仓位等级以及性别对获救情况影响 因为Survived字段是0-1(1是获救) sns.barplot图是按照y属性求均值，在此处正好是获救占比
# sns.pointplot(x='Sex',y='Survived',hue='Pclass',data=data_train)
# plt.show()

# sns.pointplot(x='Pclass',y='Survived',hue='Sex',data=data_train)
# plt.show()

# factorplot可以指定kind 来做到上边的所有图类型默认kind='point'
# sns.factorplot(x='Pclass',y='Survived',hue='Sex',data=data_train)
# sns.factorplot(x='Pclass',y='Survived',hue='Sex',data=data_train,kind='point')
# sns.factorplot(x='Pclass',y='Survived',hue='Sex',data=data_train,kind='box')
# sns.factorplot(x='Pclass',y='Survived',hue='Sex',data=data_train,kind='strip')
# sns.factorplot(x='Pclass',y='Survived',hue='Sex',data=data_train,kind='violin')
# sns.factorplot(x='Pclass',y='Survived',hue='Sex',data=data_train,kind='swarm')
# sns.factorplot(x='Pclass',y='Survived',hue='Sex',data=data_train,kind='bar')
# plt.show()

print(data_train['Sex'])
data_train['SexNum'] = data_train['Sex'].replace(['male','female'],[0,1])
print()
print(data_train[['Sex','SexNum']])

# from sklearn
