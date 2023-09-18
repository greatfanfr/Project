from __future__ import print_function
from numpy import *
import matplotlib.pyplot as plt

## 使用logistic regression 实现数据及分类，修改正则化
def load_data(file_name):
    dataMat=[]
    labelMat=[]
    fr = open(file_name)
    for line in fr.readlines():
        dataline=[]
        lineArr = line.strip().split()
        for x in range(0,len(lineArr)-2):
            dataline.append(float(lineArr[x]))
        dataMat.append(dataline)
        labelMat.append(lineArr[-1])
    print(dataMat)
    print(labelMat)
    return dataMat,labelMat
# print(load_data("data/5.Logistic/TestSet.txt"))


def sigmoid(X):
    return 1/(1+exp(-X))



def gradAscent(dataMat,labelMat):
    return
