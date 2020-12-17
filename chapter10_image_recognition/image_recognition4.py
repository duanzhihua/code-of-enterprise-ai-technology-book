# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm

digits =datasets.load_digits()
clf =svm.SVC(gamma =0.001,C=100)

X,y =digits.data[:-40],digits.target[:-40]
clf.fit(X,y)

print ("Prediction: " , clf.predict(digits.data[-42].reshape(1,-1)))

plt.imshow(digits.images[-42],cmap =plt.cm.gray_r,interpolation ='nearest')
plt.show()

