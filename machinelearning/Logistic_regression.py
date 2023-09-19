from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

## 使用logistic regression 实现数据及分类，修改正则化
def load_data(file_name):
    dataMat=[]
    labelMat=[]
    fr = open(file_name)
    for line in fr.readlines():
        dataline=[]
        lineArr = line.strip().split()
        for x in range(0,len(lineArr)-1):
            dataline.append(float(lineArr[x]))
        dataline.append(1.0)
        dataMat.append(dataline)
        labelMat.append(int(float(lineArr[-1])))
    # print(dataMat)
    # print(labelMat)
    dataMat = np.mat(dataMat)
    print("the size of x matrix: " + str(dataMat.shape))
    labelMat = np.mat(labelMat)
    print("the size of label matrix: " + str(labelMat.shape))
    return dataMat,labelMat


def sigmoid(X):
    return 1/(1+np.exp(-X))

def loss_functin(dataMat,labelMat,weight):
    J=np.dot(labelMat,np.dot(dataMat,weight))-np.log(1+np.dot(dataMat,weight))
    # J=np.dot(labelMat,np.dot(dataMat,weight))-sigmoid(np.dot(dataMat,weight))*dataMat
    loss_value=-np.average(J)
    return loss_value
def get_weights_update(dataMat,labelMat,weight):
    gradient_update=J=np.dot(labelMat,dataMat)-sigmoid(np.dot(dataMat,weight))*dataMat



def gradAscent(dataMat,labelMat):
    ###设置算法停止次数
    terminate_condition_T=400
    ###设置学习率
    learning_rate=0.01
    ###设置初始参数值 全为1
    weights=np.ones(22)

    t=0
    while t<terminate_condition_T:
        weights=weights+





    return


datamat,labelMat=load_data("data/5.Logistic/HorseColicTraining.txt")
gradAscent(datamat,labelMat)
