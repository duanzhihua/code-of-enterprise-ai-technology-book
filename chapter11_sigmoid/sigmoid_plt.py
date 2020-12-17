# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import mpl_toolkits.axisartist as axisartist
#https://blog.csdn.net/weixin_45342712/article/details/96131234
def sigmoid(x):
    # 直接返回sigmoid函数
    return 1. / (1. + np.exp(-x))
 
 
def plot_sigmoid():
    # param:起点，终点，间距
    x = np.arange(-8, 8, 0.2)
    y = sigmoid(x)
    #y = 0.15*x+3

    fig = plt.figure( )

    ax = axisartist.Subplot(fig, 1, 1, 1)
    fig.add_axes(ax)

    ax.axis[:].set_visible(False)

    #ax.axis["x"] = ax.new_floating_axis(0, 0)
    ax.axis["x"] = ax.new_floating_axis(0, -0.05)
    #ax.axis["x"] = ax.new_floating_axis(0, 0.5)
    ax.axis["y"] = ax.new_floating_axis(1, -8)
    #ax.axis["y"] = ax.new_floating_axis(1, 0)
    ax.axis["x"].set_axis_direction('top')
    ax.axis["y"].set_axis_direction('left')
    ax.axis["x"].set_axisline_style("-|>", size = 1.0)
    ax.axis["y"].set_axisline_style("-|>", size = 1.0)

    #ax.set_xlim(-4,4)
    #ax.set_ylim(0, 1)

    ax.set_xlim(-8,8)
    #ax.set_ylim(-0.0, 1.1)
    ax.set_ylim(-0.05, 1.1)
    #plt.xticks([])
    #ax.set_yticks([0, 1 ])
    ax.set_yticks([ ])
    ax.set_xticks([  ])
    plt.axhline(y=1.0, color='r',  linestyle='--')
    #plt.axhline(y=0.0, color='r', linestyle='--')

    #plt.axhline(y=0.5, color='r', linestyle='--')
    plt.axhline(y=0, color='r', linestyle='--')
    #ax.axis["x"].set_axisline_style("-|>", size=2.0)
    #ax.axis["y"].set_axisline_style("-|>", size=2.0)


    plt.plot(x, y)
    plt.show()
 
 
if __name__ == '__main__':
    plot_sigmoid()
