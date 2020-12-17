import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist

#创建画布
fig = plt.figure()
#使用axisartist.Subplot方法创建一个绘图区对象ax
ax = axisartist.Subplot(fig, 111)
#将绘图区对象添加到画布中
fig.add_axes(ax)
#通过set_axisline_style方法设置绘图区的底部及左侧坐标轴样式
#"-|>"代表实心箭头："->"代表空心箭头
ax.axis["bottom"].set_axisline_style("-|>", size = 1.5)
ax.axis["left"].set_axisline_style("->", size = 1.5)
#通过set_visible方法设置绘图区的顶部及右侧坐标轴隐藏
ax.axis["top"].set_visible(False)
ax.axis["right"].set_visible(False)
x=[1,4]
y=[1,8]
plt.plot(x,y)
plt.show()
