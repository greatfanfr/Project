# Author:阿珍一号
# date:2023/10/17
import pandas as pd
import matplotlib.pyplot as plt
import os
# print(os.path)

###  进行数据EDA
trainData = pd.read_csv("./data/train.csv",header=0)
# trainData.head()
##打印数据前10行
# print(trainData.head(10))
##打印数据的基本情况
# print(trainData.describe())