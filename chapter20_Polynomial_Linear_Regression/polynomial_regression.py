# Polynomial Regression

# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文
 
# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values #输入的Features必须以矩阵的方式存在，也是所有的AI算法中Input中数据存在的唯一方式
y = dataset.iloc[:, 2].values

# 因为需要获得P1-P10之间更精确的Salary曲线，所以我们把所有的数据都作为Training Set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4) #这里是Degree是指对Feature进行多少次方的操作
X_poly = poly_reg.fit_transform(X) #对输入的Feature进行高次方的变换，此时就变成新的Feature了，这是Polynomial Regression真正需要的Feature
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y) #此处是PolynomiaolRegression产生作用的地方

# Visualising the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('线性回归')
plt.xlabel('级别')
plt.ylabel('工资')
#plt.xlim((-0, 11))
plt.ylim((-0, 1072800))
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('多项式回归(参数设置为4)')
plt.xlabel('级别')
plt.ylabel('工资')
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('多项式回归(参数设置为4)')
plt.xlabel('级别')
plt.ylabel('工资')
plt.show()

# Predicting a new result with Linear Regression
print(lin_reg.predict(np.array([[6.5]])))

# Predicting a new result with Polynomial Regression
print(lin_reg_2.predict(poly_reg.fit_transform(np.array([[6.5]]))))