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
    print("the size of x matrix: " + str(dataMat.shape)) ### m*n
    labelMat = np.mat(labelMat)
    print("the size of label matrix: " + str(labelMat.shape)) ### 1*m
    return dataMat,labelMat


def sigmoid(x):
    # return 1/(1+np.exp(-X))
    # if inx.all() >= 0:  # 对sigmoid函数的优化，避免了出现极大的数据溢出
    #     return 1.0 / (1 + np.exp(-inx))
    # else:
    #     return np.exp(inx) / (1 + np.exp(inx))
    indices_pos = np.nonzero(x >= 0)
    indices_neg = np.nonzero(x < 0)

    y = np.zeros_like(x)
    y[indices_pos] = 1 / (1 + np.exp(-x[indices_pos]))
    y[indices_neg] = np.exp(x[indices_neg]) / (1 + np.exp(x[indices_neg]))
    return y

def loss_functin(dataMat,labelMat,weight):
    J=np.dot(labelMat,np.dot(dataMat,weight))-np.sum(np.dot(dataMat,weight))
    # J=np.dot(labelMat,np.dot(dataMat,weight))-sigmoid(np.dot(dataMat,weight))*dataMat
    loss_value=-J/len(labelMat)
    return loss_value
def get_weights_update(dataMat,labelMat,weight):
    gradient_update=np.dot(labelMat,dataMat)-np.dot(sigmoid(np.dot(dataMat,weight)).transpose(),dataMat)
    return gradient_update


def gradAscent(dataMat,labelMat):
    ###设置算法停止次数
    terminate_condition_T=400
    ###设置学习率
    learning_rate=0.01
    ###设置初始参数值 全为1
    weights=np.zeros((22,1))  ### n*1
    print("the size of label weights:"+str(weights.shape))
    t=0
    loss_value_history=[]
    t_list=[]
    while t<terminate_condition_T:
        weights=weights+learning_rate*get_weights_update(dataMat,labelMat,weights).transpose()
        if t%50==0:
           loss=loss_functin(dataMat,labelMat,weights)
           loss_value_history.append(loss[0])
           t_list.append(t)
           print("has executed "+str(t)+"times")
        t=t+1
    return loss_value_history,weights,t_list

def model_prediction(weights):

    return ;


datamat, labelMat = load_data("data/5.Logistic/HorseColicTraining.txt")
loss_value_history, weights,t_list=gradAscent(datamat,labelMat)
print(loss_value_history)
# plt.xlabel('iteration times')
# plt.ylabel('Loss_value')
# plt.plot(t_list,loss_value_history)
# plt.show()
