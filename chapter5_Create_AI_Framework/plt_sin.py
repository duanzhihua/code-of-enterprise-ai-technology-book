# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
x = np.arange(0, 5, 0.1)
y = np.sin(x)

ax = plt.gca()        
ax.spines['left'].set_position(('data', 0))
plt.xlim(0, 5)
plt.plot(x, y)

