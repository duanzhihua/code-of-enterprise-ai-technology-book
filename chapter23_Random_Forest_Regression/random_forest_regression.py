# Random Forest Regression

#导入库
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#加载数据集
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#划分训练集及测试集
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

#特征缩放
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# 在数据集上拟合 Random Forest Regression  
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators =100, random_state = 0)
regressor.fit(X, y)

#预测结果
y_pred = regressor.predict([[6.5]])

# 可视化结果 the Random Forest Regression   (高分辨率)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('随机森林回归(参数为100)')
plt.xlabel('级别')
plt.ylabel('工资')
plt.show()