# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
 
import matplotlib.pyplot as plt # For plotting.
import numpy as np # To create matrices.
def f(x):
    """ReLU returns 1 if x>0, else 0."""
    return np.maximum(0,x)
 
print ("f(1): ",f(1))
print ("f(3): ",f(3))
print ("f(-1): ",f(-1))
print ("f(-3): ",f(-3))
 
print ("Relu Function:  ")
X = np.arange(-4,5,1)
print (X)
 
Y = f(X)
# All negative values are 0.
print (Y)
 
plt.plot(X,Y,'o-')
plt.ylim(-1,5)
plt.grid()
 
plt.xlabel('$x$', fontsize=22)
plt.ylabel('$f(x)$', fontsize=22)
plt.figure(figsize=(7,7))
 
X_neg = np.arange(-4,1,1) # Negative numbers.
plt.plot(X_neg,f(X_neg),'.-', label='$f\'(x) =0$'); # Plot negative x, f(x)
X_pos = np.arange(0,5,1) # Positive numbers
plt.plot(X_pos, f(X_pos), '.-g',label='$f\'(x)=1$') # Plot positive x, f(x)
plt.plot(0,f(0),'or',label='$f \'(x)=$undefined but set to 0') # At 0.
 
plt.ylim(-1,5)
plt.grid()
plt.xlabel('$x$', fontsize=22)
plt.ylabel('$f(x)$', fontsize=22) # Make plot look nice.
plt.legend(loc='best', fontsize=16)
