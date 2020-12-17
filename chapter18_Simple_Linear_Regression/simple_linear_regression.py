# Simple Linear Regression

#数据（预）处理的三大神器：Numpy，Pandas、Matplotlib
#Numpy： Python中数据处理最流行和最强大的库（之一），尤其是对矩阵进行了全面的支持；
#Pandas：以Table的方式对数据进行处理
#Matplotlib： 对开发者最为友好的数据可视化工具之一
#上述三个Libraries是老师在开发所有的AI项目（Machine Learning, Deep Learning和Reinforcement Learning）均有使用

#第一步：导入上面的三个Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#解决中文显示问题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False


#第二步：导入源数据Data.csv
dataset = pd.read_csv('Salary_Data.csv')

#第三步 把数据切分为Independent Variables和Depedent Variable
#逗号之前代表多少Rows的信息，如果是“：”则表示取所有行；逗号之后代表Columns的信息，如果是“：”则代表所有的列
#“：”之前的数代表Rows或者Columns的索引，在Pandas中该索引是从0开始的，如果“：”之后使用-1则表示不取最后一行或一列
#如果在没有“：”则表示只取这个索引的值
X = dataset.iloc[:, :-1].values #Indepeendent Variable

y = dataset.iloc[:, 1].values #Dependent Variable

#第四步： 把数据分为Training&Testing Set
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""


from sklearn.linear_model import LinearRegression

#第五步：通过使用机器学习库来实例化Model，这里的Model是LinearRegression
regressor = LinearRegression()

#第六步：进行训练，具体训练的方式如下：
# 把我们的Training Data放进sklearn的LinearRegression中进行fitting，得出最实际的参数的值
# 也就是说要根据输入的Years和Salaries之间的关系找出最接近实际值的y = ax + b中a和b的值
regressor.fit(X_train, y_train)

#第七步： 根据fitting好的y = ax + b来进行预测
y_pred = regressor.predict(X_test)

# 第八步：通过第三方可视化库matplotlib来可视化我们的训练的准确度
plt.scatter(X_train, y_train, color = 'red') #描绘出每一个工作年限和工资之间的坐标构成的点
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('工资及工作年限 (训练集)')
plt.xlabel('工作年限/年')
plt.ylabel('工资')
plt.xlim((0, 10.8))
plt.show()

# 第九步： 进行预测并显示预测的值和实际测试的真实的值之间的关系
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue') 
plt.title('工资及工作年限(测试集)')
plt.xlabel('工作年限/年')
plt.ylabel('工资')
plt.xlim((0, 11))
plt.show()