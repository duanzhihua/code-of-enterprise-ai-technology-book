# Multiple Linear Regression

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
dataset = pd.read_csv('50_Startups.csv')

#第三步 把数据切分为Independent Variables和Depedent Variable
#逗号之前代表多少Rows的信息，如果是“：”则表示取所有行；逗号之后代表Columns的信息，如果是“：”则代表所有的列
#“：”之前的数代表Rows或者Columns的索引，在Pandas中该索引是从0开始的，如果“：”之后使用-1则表示不取最后一行或一列
#如果在没有“：”则表示只取这个索引的值
X = dataset.iloc[:, :-1].values #Indepeendent Variable
y = dataset.iloc[:, 4].values #Dependent Variable

#第四步：对Categorical类型的Feature进行编码，这里是对State进行编码
#首先使用LabelEncoder把California,Florida,New York编码成为0，1，2
#然后使用OneHotEncoder进行进一步的扁平化处理，也就是说California,Florida,New York并没有大小之分
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# 第五步：去掉对结果基本没有影响或者完全没有影响的Feature
X = X[:, 1:] #去掉了第一个Feature

#第六步： 把数据分为Training&Testing Set
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#第七步：通过使用机器学习库来实例化Model，这里的Model是LinearRegression
from sklearn.linear_model import LinearRegression

#第八步：进行训练，具体训练的方式如下：
# 把我们的Training Data放进sklearn的LinearRegression中进行fitting，得出最近接实际的参数的值
# 也就是说要根据输入的Years和Salaries之间的关系找出最接近实际值的y = ax + b中a和b的值
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)


import statsmodels.formula.api as sm

#在数据源的最前面加入一个Bias = 1，是按照列进行组拼，一共50行一列
X = np.append(arr = np.ones((50,1)).astype(int),
              values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)