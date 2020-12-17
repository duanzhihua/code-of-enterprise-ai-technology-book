# -*- coding: utf-8 -*-
# !/usr/bin/python
import torch
from torch import nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文
plt.rcParams['axes.unicode_minus']=False  #显示负号
x_data = pd.read_csv('./data/deeplearn_data/x_data/part-00000', header=None)
y_data = pd.read_csv('./data/deeplearn_data/y_data/part-00000', header=None)
#x_data = pd.read_csv('/mnt/spark_alluxio_fuse/x_data/part-00000', header=None)
#y_data = pd.read_csv('/mnt/spark_alluxio_fuse/y_data/part-00000', header=None)

x_data.columns = ['x0', 'x1']
y_data.columns = ['label']

#x_data = x_data.sort_values('x0',ascending =false)
#y_data = y_data.sort_values('label',ascending =false)
x_n = x_data[:].values.astype(np.float32)
y_n = y_data[:].values.astype(np.float32)
x = torch.from_numpy(x_n)
y = torch.from_numpy(y_n)


class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.lr = nn.Linear(2, 1)
        self.sm = nn.Sigmoid()

    def forward(self, x):
        x = self.lr(x)
        x = self.sm(x)
        return x


logistic_model = LogisticRegression()
if torch.cuda.is_available():
    logistic_model.cuda()

# 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(logistic_model.parameters(), lr=1e-3, momentum=0.9)

# 开始训练
for epoch in range(1000):
    if torch.cuda.is_available():
        x_data = Variable(x).cuda()
        y_data = Variable(y).cuda()
    else:
        x_data = Variable(x)
        y_data = Variable(y)

    out = logistic_model(x_data)
    loss = criterion(out, y_data)
    print_loss = loss.data.item()
    mask = out.ge(0.5).float()  # 以0.5进行分类
    correct = (mask == y_data).sum()  # 计算正确个数
    acc = correct.item() / x_data.size(0)  # 计算精度
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    #  打印日志
    if (epoch + 1) % 20 == 0:
        print('*' * 20)
        print('epoch {}'.format(epoch + 1))
        print('loss is {:.4f}'.format(print_loss))
        print('acc is {:.4f}'.format(acc))

# 结果可视化
w0, w1 = logistic_model.lr.weight[0]
w0 = float(w0.item())
w1 = float(w1.item())
b = float(logistic_model.lr.bias.item())
plot_x = np.arange(-7, 7, 0.1)
plot_y = (-w0 * plot_x - b) / w1
plt.scatter(x.data.numpy()[0:99, 0], x.data.numpy()[0:99, 1],   marker='+', color='red', label='类别为0')
plt.scatter(x.data.numpy()[100:199, 0], x.data.numpy()[100:199, 1],  marker='o', color='green', label='类别为1')
plt.legend(loc=(0.05, 0.82)) 

plt.title('逻辑回归')
plt.xlabel('X0')
plt.ylabel('X1')

plt.plot(plot_x, plot_y)
plt.xlim(-0.5, 1.5)
plt.ylim(-1.5, 2)
plt.savefig("LogisticRegression.png")
plt.show()
