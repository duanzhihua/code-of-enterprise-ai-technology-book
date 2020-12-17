# -*- coding: utf-8 -*-
#https://computational-communication.com/pytorch_lstm_time_series/
from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 51)
        self.lstm2 = nn.LSTMCell(51, 51)
        self.linear = nn.Linear(51, 1)

    def forward(self, input, future = 0):#input(97,999)
        outputs = []
        h_t = torch.zeros(input.size(0), 51, dtype=torch.double)#size(97,51)
        c_t = torch.zeros(input.size(0), 51, dtype=torch.double)#size(97,51)
        h_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)#size(97,51)
        c_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)#size(97,51)
        #chunk在给定维度(轴)上将输入张量进行分块
        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):# 999  (97,1)
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))#input_t (97,1)
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))#h_t2 （97，51） c_t2（97，51）
            output = self.linear(h_t2)
            outputs += [output] #999 行（97，1）预测999行 （3，1）
        for i in range(future):# if we should predict the future
            #h_t （3，51）  , c_t （3，51）
            h_t, c_t = self.lstm1(output, (h_t, c_t)) #预测output（3，1）
            #h_t2,（3，51） c_t2（3，51）
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)# 预测（3，1）
            outputs += [output] #（1001行 （3，1））
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs #（97，999）


if __name__ == '__main__':
    # set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0)
    # load data and make training set
    data = torch.load('traindata.pt')
    input = torch.from_numpy(data[3:, :-1])
    target = torch.from_numpy(data[3:, 1:])
    test_input = torch.from_numpy(data[:3, :-1])
    test_target = torch.from_numpy(data[:3, 1:])
    # build the model
    seq = Sequence()
    seq.double()
    criterion = nn.MSELoss()
    # use LBFGS as optimizer since we can load the whole data to train
    optimizer = optim.LBFGS(seq.parameters(), lr=0.8)
    #begin to train
    for i in range(15):
        print('STEP: ', i)
        def closure():
            optimizer.zero_grad()
            out = seq(input)#（97，999）
            loss = criterion(out, target)
            print('loss:', loss.item())
            loss.backward()
            return loss
        optimizer.step(closure)
        # begin to predict, no need to track gradient here
        with torch.no_grad():
            future = 1000
            pred = seq(test_input, future=future) # pred  （3，1999）test_input（3，999）
            loss = criterion(pred[:, :-future], test_target)#test_target （3，999）
            print('test loss:', loss.item())
            #detach切断梯度的反向传播
            y = pred.detach().numpy()# 3个记录 （0，1999）
        # draw the result
        plt.figure(figsize=(30,10))
        plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
        plt.xlabel('x', fontsize=20)
        plt.ylabel('y', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        def draw(yi, color):
            plt.plot(np.arange(input.size(1)), yi[:input.size(1)], color, linewidth = 2.0)
            plt.plot(np.arange(input.size(1), input.size(1) + future), yi[input.size(1):], color + ':', linewidth = 2.0)
        draw(y[0], 'r')
        draw(y[1], 'g')
        draw(y[2], 'b')
        plt.savefig('predict%d.pdf'%i)
        plt.close()