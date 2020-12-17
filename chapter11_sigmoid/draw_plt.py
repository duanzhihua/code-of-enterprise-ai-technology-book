# encoding=utf-8
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import mpl_toolkits.axisartist as axisartist
plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams.update({'font.size': 8})
x = range(11)
y = range(11)



fig = plt.figure( )

ax = axisartist.Subplot(fig, 1, 1, 1)
fig.add_axes(ax)
ax.axis[:].set_visible(False)

#ax.axis["x"] = ax.new_floating_axis(0, 0)
ax.axis["x"] = ax.new_floating_axis(0, 0.01)
#ax.axis["x"] = ax.new_floating_axis(0, 0.5)
ax.axis["y"] = ax.new_floating_axis(1, 0.01)
ax.axis["x"].set_axis_direction('top')
ax.axis["y"].set_axis_direction('left')
ax.axis["x"].set_axisline_style("-|>", size=1.0)
ax.axis["y"].set_axisline_style("-|>", size=1.0)
ax.set_xlim(0, 10)
# ax.set_ylim(-0.0, 1.1)
ax.set_ylim(0, 10)
#ax.set_yticks(['20%','80%'])
#ax.set_xticks(['20%','80%'])
def to_percent(temp, position):
     return '%1.0f' % (10 * temp) + '%'


plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
plt.gca().xaxis.set_major_formatter(FuncFormatter(to_percent))
plt.plot(x, y, color='white')
plt.tick_params(labelsize=23)
labels = ax.get_xticklabels() + ax.get_yticklabels()
# print labels
[label.set_fontname('Times New Roman') for label in labels]
 
plt.show()