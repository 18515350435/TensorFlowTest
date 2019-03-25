import pandas as pd #数据分析
import numpy as np #科学计算


data_train = pd.read_csv("Train.csv")
# print(data_train.info())
# 1.先整体观察一下数据，发现有不少列存在空值：Age Cabin Embarked
# data_train.info()

# 2.在 titanic数据分析.py 中了解到了数据中属性间对能否获救的关系
    # a.女性比男性获救几率大很多
    # b.小孩比大人获救几率大很多，但经验猜测不是与age成线性关系
    # c.有Cabin记录的似乎获救概率稍高一些
    # d.Pclass的仓位级别越高获救几率越大，也许与当时的人的社会地位和拥有财富相关
    # e.兄弟姐妹越多不被获救几率稍微偏低【也许是有人可替代传宗接代】
    # e.父母孩子数量越多获救几率稍微偏高【也许是跟需要人照顾】
    # f.票价发现票价越高获救概率大
    # g.登船港口不同获救概率明显不一样C>Q>S
    # h.Ticket票码是否是纯数字的对于获救并无明显影响


# 读取训练集并将数据进行预处理

# ---------------------部分属性进行有必要的数值填充，部分属性数值化或行离散因子化-----------------------------------


# 处理年龄缺失值问题
from sklearn.ensemble import RandomForestRegressor

# 使用 RandomForestClassifier 填补缺失的年龄属性
def set_missing_ages(df):
    # 将男女用01代替0男 1女
    df['Sex_'] = df['Sex'].replace(['male', 'female'], [0, 1])
    # 将是否有船舱记录用01代替  0有  1没有
    df['Cabin_'] = df['Cabin'].isnull().astype('int');
    # print(df[['Sex','Cabin','Sex_','Cabin_']])

    # 认为称呼中Mr. Mrs. Miss. 不同称呼人群的平均年龄不一样 为了让称呼也作为补充缺失年龄所参考特征值 所以追加如下
    df['Name_1'] = 0
    df['Name_2'] = 0
    df['Name_3'] = 0
    df.loc[df.Name.str.contains('Mr\.'), 'Name_1'] = 1
    df.loc[df.Name.str.contains('Mrs\.'), 'Name_2'] = 1
    df.loc[df.Name.str.contains('Miss\.'), 'Name_3'] = 1

    if len(df[df.Fare.isnull()])>0:
        # -------------------------------------------------------
        # 把已有的数值型特征取出来丢进Random Forest Regressor中
        Fare_df = df[['Fare', 'Name_1', 'Name_2', 'Name_3',  'Parch', 'SibSp', 'Pclass']]

        # 乘客分成已知年龄和未知年龄两部分
        known_Fare = Fare_df[Fare_df.Fare.notnull()].as_matrix()
        unknown_Fare = Fare_df[Fare_df.Fare.isnull()].as_matrix()

        # y即目标年龄
        y = known_Fare[:, 0]

        # X即特征属性值
        X = known_Fare[:, 1:]

        # fit到RandomForestRegressor之中
        rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
        rfr.fit(X, y)

        # 用得到的模型进行未知年龄结果预测
        predictedFares = rfr.predict(unknown_Fare[:, 1::])

        # 用得到的预测结果填补原缺失数据
        df.loc[(df.Fare.isnull()), 'Fare'] = predictedFares
        # -----------------------------------------
    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Age','Name_1', 'Name_2', 'Name_3', 'Fare', 'Parch', 'SibSp', 'Pclass']]

    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    # y即目标年龄
    y = known_age[:, 0]

    # X即特征属性值
    X = known_age[:, 1:]

    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1::])

    # 用得到的预测结果填补原缺失数据
    df.loc[(df.Age.isnull()), 'Age'] = predictedAges

    # 根据年龄将人员分组（不同年龄段的趋势不一样）
    # df['Age_B'] = 0
    # df['Age_M'] = 0
    # df['Age_S'] = 0
    # df.loc[(df.Age > 34), 'Age_B'] = 1
    # df.loc[(df.Age > 16) & (df.Age <= 34), 'Age_M'] = 1
    # df.loc[(df.Age <= 16), 'Age_S'] = 1

    # 因为逻辑回归建模时，需要输入的特征都是数值型特征，我们通常会先对类目型的特征因子化。
    # 暂时只认为另外的非数值列Embarked 会对预测结果有影响，所以将其进行离散因子化
    dummies_Embarked = pd.get_dummies(df['Embarked'], prefix='Embarked')
    df = pd.concat([df, dummies_Embarked], axis=1)
    dummies_Pclass = pd.get_dummies(df['Pclass'], prefix='Pclass')
    df = pd.concat([df, dummies_Pclass], axis=1)
    # 去除非数值型的列以及已经无用的'Name_',他只是用来修正缺失Age用的
    df.drop([ 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked','Pclass', 'Name_1', 'Name_2', 'Name_3'], axis=1, inplace=True)
    # ---------------特征缩放-----------------------------
    # 打印df发现属性中的发现Age和Fare数值幅度变化太大
    # 逻辑回归与梯度下降中如果各属性值之间scale差距太大，将对收敛速度造成几万点伤害值！甚至不收敛！
    # 显示所有列
    pd.set_option('display.max_columns', None)
    # 显示所有行
    # pd.set_option('display.max_rows', None)
    # 设置value的显示长度为100，默认为50
    pd.set_option('max_colwidth', 100)
    # print(df)
    # 用scikit-learn里面的preprocessing模块对Age和Fare这俩货做一个scaling，所谓scaling，其实就是将一些变化幅度较大的特征化到[-1,1]之内
    import sklearn.preprocessing as preprocessing
    scaler = preprocessing.StandardScaler()

    age_scale_param = scaler.fit(df[['Age']])
    df['Age_scaled'] = scaler.fit_transform(df[['Age']], age_scale_param)
    fare_scale_param = scaler.fit(df[['Fare']])
    df['Fare_scaled'] = scaler.fit_transform(df[['Fare']], fare_scale_param)
    return df, rfr


# def set_Cabin_type(df):
#     df.loc[(df.Cabin.notnull()), 'Cabin'] = "Yes"
#     df.loc[(df.Cabin.isnull()), 'Cabin'] = "No"
#     return df

# df = set_Cabin_type(df)

df, rfr = set_missing_ages(data_train)
# 用正则取出我们要的属性值
train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_|Sex_|Pclass_.*|Embarked_.*')
train_np = train_df.as_matrix()
# print(train_np)



# --------------------逻辑回归建模-----------------------
from sklearn import linear_model

# y即Survival结果
y = train_np[:, 0]

# X即特征属性值
X = train_np[:, 1:]

# fit到RandomForestRegressor之中
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf.fit(X, y)
# 输出模型
# print(clf)

# -----------------------test.csv直接丢进model里---------------------------
data_test = pd.read_csv("test.csv")
# print(data_test.info())
df_test, rfr_test = set_missing_ages(data_test)
# # 用正则取出我们要的属性值
test = df_test.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_|Sex_|Pclass_.*|Embarked_.*')
predictions = clf.predict(test)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})
result.to_csv("predictions_result.csv", index=False)

# -------------------------交叉验证--------------------------------------------
# from sklearn import cross_validation
# 改为下面的从model_selection直接import cross_val_score 和 train_test_split
from sklearn.model_selection import cross_val_score, train_test_split

 #简单看看打分情况
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
all_data = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
X = all_data.as_matrix()[:,1:]
y = all_data.as_matrix()[:,0]
# print(cross_validation.cross_val_score(clf, X, y, cv=5))
print(cross_val_score(clf, X, y, cv=5))

# ---------------------------------------------------把交叉验证里面的bad case拿出来看看，看看人眼审核，是否能发现什么蛛丝马迹，
# ---------------------------------------------------是我们忽略了哪些信息，使得这些乘客被判定错了。再把bad case上得到的想法和前头系数分析的合在一起，然后逐个试试
# 分割数据，按照 训练数据:cv数据 = 7:3的比例
# split_train, split_cv = cross_validation.train_test_split(df, test_size=0.3, random_state=0)
split_train, split_cv = train_test_split(df, test_size=0.3, random_state=42)

train_df = split_train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_|Sex_|Pclass_.*|Embarked_.*')

# 生成模型
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf.fit(train_df.as_matrix()[:,1:], train_df.as_matrix()[:,0])

# 对cross validation数据进行预测

cv_df = split_cv.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_|Sex_|Pclass_.*|Embarked_.*')
predictions = clf.predict(cv_df.as_matrix()[:,1:])
print(predictions)
origin_data_train = pd.read_csv("train.csv")
bad_cases = origin_data_train.loc[origin_data_train['PassengerId'].isin(split_cv[predictions != cv_df.as_matrix()[:,0]]['PassengerId'].values)]
bad_cases.to_csv("bad_case_result.csv", index=False)


# -------------------------------------学习曲线图-------------------------------------------
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['font.sans-serif'] = ['SimHei']#防止绘图中的中文乱码
# from sklearn.learning_curve import learning_curve  修改以fix learning_curve DeprecationWarning
from sklearn.model_selection import learning_curve


# 用sklearn的learning_curve得到training_score和cv_score，使用matplotlib画出learning curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1,
                        train_sizes=np.linspace(.05, 1., 20), verbose=0, plot=True):
    """
    画出data在某模型上的learning curve.
    参数解释
    ----------
    estimator : 你用的分类器。
    title : 表格的标题。
    X : 输入的feature，numpy类型
    y : 输入的target vector
    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
    n_jobs : 并行的的任务数(默认1)
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    if plot:
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel(u"训练样本数")
        plt.ylabel(u"得分")
        plt.gca().invert_yaxis()
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                         alpha=0.1, color="b")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
                         alpha=0.1, color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"训练集上得分")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label=u"交叉验证集上得分")

        plt.legend(loc="best")

        plt.draw()
        plt.gca().invert_yaxis()
        plt.show()

    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
    return midpoint, diff


plot_learning_curve(clf, u"学习曲线", X, y)

# ----------------------------模型融合------------------------
# 因为暂时只学了logistic regression逻辑回归模型，所以使用BaggingRegressor来做单种类模型，多训练集，来做单种类多的多个模型来实现模型融合
from sklearn.ensemble import BaggingRegressor

train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass.*|Mother|Child|Family|Title')
train_np = train_df.as_matrix()

# y即Survival结果
y = train_np[:, 0]

# X即特征属性值
X = train_np[:, 1:]

# fit到BaggingRegressor之中 LogisticRegression函数详解：https://blog.csdn.net/sun_shengyun/article/details/53811483
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
bagging_clf = BaggingRegressor(clf, n_estimators=20, max_samples=0.8, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=-1)
bagging_clf.fit(X, y)

test = df_test.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_|Sex_|Pclass_.*|Embarked_.*')
predictions = bagging_clf.predict(test)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})
result.to_csv("logistic_regression_bagging_predictions.csv", index=False)


