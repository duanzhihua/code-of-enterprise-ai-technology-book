# -*- coding: utf-8 -*-
#数据（预）处理的三大神器：Numpy，Pandas、Matplotlib
#Numpy： Python中数据处理最流行和最强大的库（之一），尤其是对矩阵进行了全面的支持；
#Pandas：以Table的方式对数据进行处理
#Matplotlib： 对开发者最为友好的数据可视化工具之一
#上述三个Libraries是老师在开发所有的AI项目（Machine Learning, Deep Learning和Reinforcement Learning）均有使用

#第一步：导入上面的三个Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#第二步：导入源数据Data.csv
dataset = pd.read_csv('Data.csv') #注意，需要设置当然目录为Working Directory，并把数据文件放在当前目录

#第三步：Split the data into independent variables and dependent variable
X = dataset.iloc[:,:-1].values #这是一个二维数组的Table，我们去其每一行的前三列（Features）
y = dataset.iloc[:,3].values #取所有行的最后一列（是否购买的真实的值）

#第四步：修复数据源中缺失的元素内容，我们需要修复Age和Salary所在的列
# 数据修复的目的：缺失的数据不影响我们正常的AI
#具体的修复方法：求该缺失数据所在列的平均值，把该平均值赋值给该缺失的内容
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN",strategy="mean", axis=0)
imputer = imputer.fit(X[:,1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

#第五步：因为AI只能处理数据，所以我们需要把Categorical Data转换成数字
#具体方法：打上Label

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#把Country编码成为0,1,2
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])

#进行Dummy Encoding，把Country编码成为[0,0,1]或者[0,1,0]或者[1,0,0]
enc = OneHotEncoder(categorical_features=[0])
X = enc.fit_transform(X).toarray()

#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
#把数据分为Training&Testing Set
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

#因为Salary从数字上讲远远比Age的数字大，所以需要进行Feature Scaling

from sklearn.preprocessing import StandardScaler

standardScaler = StandardScaler()
X_train = standardScaler.fit_transform(X_train)
X_test = standardScaler.transform(X_test)






