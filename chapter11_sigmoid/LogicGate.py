# -*- coding: utf-8 -*-

import numpy as np
 
 
def AND(x1, x2):
    x = np.array([1, x1, x2])
    w = np.array([-1.5, 1, 1])
    y = np.sum(w*x)
    if y <= 0:
        return 0
    else:
        return 1
 
def OR(x1, x2):
    x = np.array([1, x1, x2])
    w = np.array([-0.5, 1, 1])
    y = np.sum(w*x)
    if y <= 0:
        return 0
    else:
        return 1
 
def NAND(x1, x2):
    x = np.array([1, x1, x2])
    w = np.array([1.5, -1, -1])
    y = np.sum(w*x)
    if y <= 0:
        return 0
    else:
        return 1
 
 
if __name__ == '__main__':
    input = [(0, 0), (1, 0), (0, 1), (1, 1)]
 
    print("AND")
    for x in input:
        y = AND(x[0], x[1])
        print(str(x) + " -> " + str(y))
 
    print("OR")
    for x in input:
        y = OR(x[0], x[1])
        print(str(x) + " -> " + str(y))
 
    print("NAND")
    for x in input:
        y = NAND(x[0], x[1])
        print(str(x) + " -> " + str(y))
