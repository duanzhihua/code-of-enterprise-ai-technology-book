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



#第五步：因为AI只能处理数据，所以我们需要把Categorical Data转换成数字
#具体方法：打上Label

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#把Country编码成为0,1,2
labelencoder_X = LabelEncoder()
X[:,0 ] = labelencoder_X.fit_transform(X[:,0 ])
X[:,1 ] = labelencoder_X.fit_transform(X[:,1 ])
print(X)

#进行Dummy Encoding，把Country编码成为[0,0,1]或者[0,1,0]或者[1,0,0]
enc = OneHotEncoder(categorical_features=[0,1])
X = enc.fit_transform(X).toarray()
print(X)



