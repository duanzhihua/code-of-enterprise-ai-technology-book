import numpy as np

# sigmoid 函数
def nonlin(x,deriv=False):
    if(deriv==True): #在反向传播的时候使用，来判断哪些特征对当前的误差结果负有更大的责任
        return x*(1-x)
    return 1/(1+np.exp(-x)) #在前向传播算法中的激活函数
    
# 凡是用Numpy定义的二维数组都是矩阵,但是你也可以使用Numpy声明最简单的一维数组。
# X是我们输入的数据集，里面包含了4行数据，每一行数据都是一个训练数据
X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])
    
# 真实的输出值，因为我们特征一共有4条数据，所以这里使用矩阵的转置操作，这样就把行变成了列
# 输出值，这里是训练集对应的输出结果的比较值，也就是说我们使用y中的相应的每一个值来作为评判标准。
y = np.array([[0,0,1,1]]).T

 

# 输入随机数进行计算
np.random.seed(1) #每次的随机值都是一样的，方便学习观察

# 初始化权重，必须是矩阵，因为输入层中的每一个神经元中包含的元素值都是矩阵X中的当前训练的一条数据中的一列
#这里的*是指把2乘以随机产生的三行一列的矩阵中的每一个元素
#这里的-是指把这个三行一列的矩阵中的每一个元素都减1
weights = 2 * np.random.random((3,1)) - 1 
 
for iter in range(10000): #这里的10000是指10000个Epoch

    # 前向传播
    layer0 = X #layer0是整个神经网络的第一层，也就是输入层
    layer1 = nonlin(np.dot(layer0,weights)) #隐藏层，进行点乘操作，第一个隐藏层有3个神经元

    # 误差是多少?
    l1_error = y - layer1 #求出每一个训练值的误差

    #  用第1层中的值乘以我们sigmoid 函数的斜率。
    l1_delta = l1_error * nonlin(layer1,True)

    # 更新权重
    weights += np.dot(layer0.T,l1_delta) #三行四列乘以四行一列就变成了三行一列，而这个和我们初始化后的Weights的三行一列是一致的，所以可以进行矩阵的加法操作
    

print ("Output After Training:")
print (layer1)
