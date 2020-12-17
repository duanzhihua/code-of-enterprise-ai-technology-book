# -*- coding: utf-8 -*-
 # -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x,derive =False):
    if not derive:
        return 1 /(1+np.exp(-x))
    else:
        return x*(1-x)


def relu(x,derive =False ):
    if not derive:
        return np.maximum(0,x)
    return (x>0).astype(float)


#nonline = sigmoid
nonline = relu
X = np.array([[0,0,1],[0,1,1], [1,0,1],[1,1,1,]])
y=np.array([[0],[1], [1], [0]])


#weight
np.random.seed(1)


w1 = 2* np.random.random((3,4)) -1
b1 = 0.1* np.ones((4,))
w2 = 2* np.random.random((4,1)) -1
b2 = 0.1 * np.ones((1,))


adaptive_loss_log = []
training_times =6000
for i in range(training_times):
    A1 =np.dot(X,w1)+b1
    Z1=nonline(A1)


    A2 =np.dot(Z1,w2)+b2
    _y = Z2 = nonline(A2)
    cost = _y - y

    print('Cost :{}'.format(np.mean(np.abs(cost))))
    #calc delta
    delta_A2 = cost * nonline (Z2,derive =True)
    delta_b2 = delta_A2.sum(axis =0)
    delta_w2 = np.dot(Z1.T,delta_A2)
    delta_A1 = np.dot(delta_A2,w2.T) *nonline(Z1,derive =True)
    delta_b1 = delta_A1.sum(axis =0)
    delta_w1 = np.dot(X.T,delta_A1)

    rate=0.1
    w1 -= rate * delta_w1
    b1 -= rate * delta_b1
    w2 -= rate  * delta_w2
    b2 -= rate * delta_b2

    if i % 100 == 0:
            loss= np.mean(np.abs(cost))
            print("Epoch " + str(i), "\t", loss)
            adaptive_loss_log.append(loss)

else:
    print('Output:')
    print(_y)

#plt.plot(adaptive_loss_log, label = " sigmoid Loss Log:") #传进要进行可视化的数据集
plt.plot(adaptive_loss_log, label = " relu Loss Log:") #传进要进行可视化的数据集
plt.legend(bbox_to_anchor = (1,1), bbox_transform=plt.gcf().transFigure)

plt.show() #显示可视化结果     

