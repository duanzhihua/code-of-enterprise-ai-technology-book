# coding: utf-8
# 导入库
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文
plt.rcParams['axes.unicode_minus']=False  #显示负号
 
# 加载波士顿房价数据集
boston = datasets.load_boston(  )
#boston = datasets.load_boston( return_X_y=True)
print(boston.data.shape) 
# 定义列名
columns = "crim zn indus chas nox rm age dis rad tax ptratio b lstat".split()
# 加载sklearn数据集转换为 pandas的dataframe
x = pd.DataFrame(boston.data, columns = columns)
y = pd.DataFrame(boston.target)
 

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
 

show_steps = True

#使用正向选择，从没有预测器开始 
included = []
r2_list = []
adjusted_r2_list = []
# 跟踪模型和参数
best = {"feature":"",
        "r2":0,
        "adjusted_r2":0}
#创建线性模型
model = LinearRegression()
# 测试集的记录数
n = x_test.shape[0]

while True:
    changed = False
    if show_steps:
        print("")
    
    excluded = list(set(x.columns)-set(included))
    
    if show_steps:
        print("(步骤) 排除的特征集 = %s" % "," .join(excluded))
        
    for new_column in excluded:
        if show_steps:
            print("(步骤) 选择特征 %s  特征集 = %s " %( new_column ,  ", " .join(included + [new_column])))
 
    
        fit = model.fit(x_train[included + [new_column]], y_train)
        r2 = model.score(x_test[included + [new_column]], y_test)
        
        #模型选择的特征数
        p = len(included) + 1
        adjusted_r2 = 1 - (((1-r2)*(n-1)) / (n-p-1))
        
        if show_steps:
            print("(步骤) 本次改进的拟合度:   = %.3f; 改进的拟合度的最大值 = %.3f" 
                  % (adjusted_r2, best["adjusted_r2"]))
        
        if adjusted_r2 > best["adjusted_r2"]:
            best = {"feature":new_column,
                    "r2":r2,
                    "adjusted_r2":adjusted_r2}
            changed = True
            if show_steps:
                print("(步骤) 更新改进的拟合度的最大值: 特征 = %s; 拟合度 = %.3f; 改进的拟合度 = %.3f"
                      %(best["feature"], best["r2"], best["adjusted_r2"]))
    if changed:
        r2_list.append(best["r2"])
        adjusted_r2_list.append(best["adjusted_r2"])
        included.append(best["feature"])
        excluded = list(set(excluded) - set(best["feature"]))
        print("增加特征 %-4s 拟合度 = %.3f 改进的拟合度 = %.3f" % 
              (best["feature"], best["r2"], best["adjusted_r2"]))
    else:
        break
print("")
print("最终选择的特征")
print(", ".join(included))
 

# 绘制每个指标
x_range = len(np.array(r2_list))
plt.plot(range(1,x_range+1), r2_list,  "--", label = "拟合度")
plt.plot(range(1,x_range+1), adjusted_r2_list,"-", label = "改进的拟合度")
# 可视化
plt.xlabel("特征数")
plt.legend()
# 输出图表
plt.show()

