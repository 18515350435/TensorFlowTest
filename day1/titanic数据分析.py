import inline as inline
import pandas as pd #数据分析
import seaborn as sns
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['font.sans-serif'] = ['SimHei']#防止绘图中的中文乱码

data_train = pd.read_csv("Train.csv")

# # #兄弟姐妹数量与获救情况分析（兄弟姐妹越多不被获救几率稍微偏低【也许是有人可替代】，不一定最符合正相关）
# AF = data_train[['SibSp',"Survived"]]#只要部分列的数据
# sns.regplot(x="SibSp",y="Survived",data=AF,y_jitter=0.1,x_jitter=0.3,marker='.',logistic=True)#x_jitter,y_jitter指定在x轴或y轴上的浮动防止点位覆盖
# plt.show()
#
# # #父母孩子数量与获救情况分析（父母孩子数量越多获救几率稍微偏高【也许是跟需要人照顾】，不一定最符合正相关）
# sns.regplot(x="Parch",y="Survived",data=data_train,y_jitter=0.1,x_jitter=0.3,marker='.',logistic=True)
# plt.show()

# #年龄与获救情况分析（年龄越小获救几率越高，不一定最符合正相关）
# AF = data_train[(data_train.Age<=16)]#只要部分列的数据
# sns.regplot(x="Age",y="Survived",data=AF,y_jitter=0.1,x_jitter=0.3,marker='.',logistic=True)#x_jitter,y_jitter指定在x轴或y轴上的浮动防止点位覆盖
# plt.show()
# AF = data_train[(data_train.Age>16)&(data_train.Age<=34)]#只要部分列的数据
# sns.regplot(x="Age",y="Survived",data=AF,y_jitter=0.1,x_jitter=0.3,marker='.',logistic=True)#x_jitter,y_jitter指定在x轴或y轴上的浮动防止点位覆盖
# plt.show()
# AF = data_train[(data_train.Age>34)]#只要部分列的数据
# sns.regplot(x="Age",y="Survived",data=AF,y_jitter=0.1,x_jitter=0.3,marker='.',logistic=True)#x_jitter,y_jitter指定在x轴或y轴上的浮动防止点位覆盖
# plt.show()
data_train['Age_B']=0
data_train['Age_M']=0
data_train['Age_S']=0
data_train.loc[(data_train.Age > 34),'Age_B']=1
data_train.loc[(data_train.Age>16)&(data_train.Age<=34),'Age_M']=1
data_train.loc[(data_train.Age<=16),'Age_S']=1
print(data_train[['Age','Age_B','Age_M','Age_S']])
#性别与获救情况分析（年龄越小获救几率越高，不一定最符合正相关）
# AF = data_train[['Sex',"Survived"]]#只要部分列的数据
# print(AF['Sex'].map(lambda x: x=='male'?'0':'1'))
# AF['Sex'].e
# AF['Sex']=(AF['Sex']=='female').astype('int')#将性别用01表示方便绘制离散图
# 当有好多种类是非数字表示时候，为了用regplot加x_jitter,y_jitter来看其分布
# data_train['SexNum'] = data_train['Sex'].replace(['male','female'],[0,1])
# 将数据集中Ticket里纯数字和非纯数字的区分成两类 0纯数字 1非纯数字
# data_train['Ticket_'] = data_train['Ticket'].replace([r'^[0-9]*$',r'[^\d]'],[0,1],regex=True)
# print(data_train[['Ticket','Ticket_','Survived']])
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
# import statsmodels

# # 通过观察票价发现票价越高获救概率大
# print(data_train[data_train.Fare<300])
# sns.regplot(x='Fare',y="Survived",data=data_train,marker="+",y_jitter=.1,logistic=True)
# sns.regplot(x='Fare',y="Survived",data=data_train[data_train.Fare<300],marker="+",y_jitter=.1,logistic=True)
# sns.regplot(x='Fare',y="Survived",data=data_train,y_jitter=.4,x_jitter=40,marker="+",logistic=True)
# plt.show()

# 不同的仓位等级以及性别对获救情况影响 因为Survived字段是0-1(1是获救) sns.barplot图是按照y属性求均值，在此处正好是获救占比
# sns.barplot(x='Sex',y='Survived',hue='Pclass',data=data_train)
# plt.show()

# 不同的仓位等级以及性别对获救情况影响 因为Survived字段是0-1(1是获救) sns.barplot图是按照y属性求均值，在此处正好是获救占比
# sns.pointplot(x='Sex',y='Survived',hue='Pclass',data=data_train)
# plt.show()

sns.pointplot(x='Embarked',y='Survived',hue='Sex',data=data_train)
plt.show()

# 登船港口不同获救概率明显不一样C>Q>S
# sns.pointplot(x='Embarked',y='Survived',data=data_train)
# sns.pointplot(x='Embarked',y='Survived',hue='Sex',data=data_train)
# plt.show()

# Ticket票码是否是纯数字的对于获救并无明显影响
# data_train['Ticket_'] = data_train['Ticket'].replace([r'^[0-9]*$',r'[^\d]'],[0,1],regex=True)
# print(data_train[['Ticket','Ticket_','Survived']])
# # sns.pointplot(x='Ticket_',y='Survived',data=data_train)
# sns.pointplot(x='Ticket_',y='Survived',hue='Sex',data=data_train)
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

# data_train['SexNum'] = data_train['Sex'].replace(['male','female'],[0,1])
# data_train['Ticket_']=data_train[data_train.Ticket>='A']=1
# data_train['Ticket_']=data_train['A'>data_train.Ticket]=0
# print(data_train[['Ticket','Ticket_','Survived']])
# data_train['Name_']=0
# data_train.loc[data_train.Name.str.contains('Mr\.'),'Name_']=1
# data_train.loc[data_train.Name.str.contains('Mrs\.'),'Name_']=2
# data_train.loc[data_train.Name.str.contains('Miss\.'),'Name_']=3
# print(data_train[['Sex','Name','Name_','Age']])

# dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix= 'Cabin')
# df = pd.concat([data_train, dummies_Cabin], axis=1)
# print(df)
