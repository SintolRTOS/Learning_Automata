# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 23:56:34 2019

@author: wangjingyi
"""

import numpy as np
import time
from random import choice 
import pandas as pd
import os

#定义计算共同邻居指标的方法
#define some functions to calculate some baseline index
def Cn(MatrixAdjacency):
    Matrix_similarity = np.dot(MatrixAdjacency,MatrixAdjacency)
    return Matrix_similarity

#计算Jaccard相似性指标
def Jaccavrd(MatrixAdjacency_Train):
    Matrix_similarity = np.dot(MatrixAdjacency_Train,MatrixAdjacency_Train)
    deg_row = sum(MatrixAdjacency_Train)
    deg_row.shape = (deg_row.shape[0],1)
    deg_row_T = deg_row.T
    tempdeg = deg_row + deg_row_T
    temp = tempdeg - Matrix_similarity
    Matrix_similarity = Matrix_similarity / temp
    return Matrix_similarity

#定义计算Salton指标的方法
def Salton_Cal(MatrixAdjacency_Train):
    similarity = np.dot(MatrixAdjacency_Train,MatrixAdjacency_Train)
    deg_row = sum(MatrixAdjacency_Train)
    deg_row.shape = (deg_row.shape[0],1)
    deg_row_T = deg_row.T
    tempdeg = np.dot(deg_row,deg_row_T)
    temp = np.sqrt(tempdeg)
    np.seterr(divide='ignore', invalid='ignore')
    Matrix_similarity = np.nan_to_num(similarity / temp)
    print(np.isnan(Matrix_similarity))
    Matrix_similarity = np.nan_to_num(Matrix_similarity)
    print(np.isnan(Matrix_similarity))
    return Matrix_similarity

#定义计算Katz1指标的方法
def Katz_Cal(MatrixAdjacency):
    #α取值
    Parameter = 0.01
    Matrix_EYE = np.eye(MatrixAdjacency.shape[0])
    Temp = Matrix_EYE - MatrixAdjacency * Parameter
    Matrix_similarity = np.linalg.inv(Temp)
    Matrix_similarity = Matrix_similarity - Matrix_EYE
    return Matrix_similarity

#定义计算局部路径LP相似性指标的方法
def LP_Cal(MatrixAdjacency):
    Matrix_similarity = np.dot(MatrixAdjacency,MatrixAdjacency)
    Parameter = 0.05
    Matrix_LP = np.dot(np.dot(MatrixAdjacency,MatrixAdjacency),MatrixAdjacency) * Parameter
    Matrix_similarity = np.dot(Matrix_similarity,Matrix_LP)
    return Matrix_similarity
    
#计算资源分配（Resource Allocation）相似性指标
def RA(MatrixAdjacency_Train):
    RA_Train = sum(MatrixAdjacency_Train)
    RA_Train.shape = (RA_Train.shape[0],1)
    MatrixAdjacency_Train_Log = MatrixAdjacency_Train / RA_Train
    MatrixAdjacency_Train_Log = np.nan_to_num(MatrixAdjacency_Train_Log)
    Matrix_similarity = np.dot(MatrixAdjacency_Train,MatrixAdjacency_Train_Log)
    return Matrix_similarity

#仿真随机环境一：针对活跃性的节点对
def RandomEnviromentForActive(MatrixAdjacency,i,j):
    Index = np.random.randint(1, 5)
    print(Index)
    global IndexName
    if Index == 1:
        IndexName = '相似性指标是：Jaccard Index'
        print(IndexName)
        similarity_matrix = Jaccavrd(MatrixAdjacency)
        similarity = similarity_matrix[i,j]
    elif Index == 2:
        IndexName = '相似性指标是：Salton Index'
        print(IndexName)
        similarity_matrix = Salton_Cal(MatrixAdjacency)
        similarity = similarity_matrix[i,j]
    elif Index == 3:
        IndexName = '相似性指标是：Katz Index'
        print(IndexName)
        similarity_matrix = Katz_Cal(MatrixAdjacency)
        similarity = similarity_matrix[i,j]
    else:
        IndexName = '相似性指标是：RA Index'
        print(IndexName)
        similarity_matrix = RA(MatrixAdjacency)
        similarity = similarity_matrix[i,j]
    return similarity     
    
#随机环境二：主要针对非活跃性的节点对
def RandomEnviromentForNonActive():

    Action = np.random.randint(1, 4)
    if Action == 1:
        ActionName = 'ID3'
        similarity_matrix = ID3_Cal(MatrixAdjacency)
        #similarity = similarity_matrix[i,j]
    elif Action == 2:
        ActionName = 'CART'
        similarity_matrix = Cart_Cal(MatrixAdjacency)
        #similarity = similarity_matrix[i,j]
    elif Action == 3:
        ActionName = 'C4.5'
        similarity_matrix = C4_Cal(MatrixAdjacency)
        #similarity = similarity_matrix[i,j]
    return similarity

#构建学习自动机的智能体(To Construct the agent)
def ContructionAgent(filepath,n1,n2):
    f = open(filepath)
    lines = f.readlines()
    A = np.zeros((50, 50), dtype=float)
    A_row = 0
    for line in lines:
        list = line.strip('\n').split(' ')
        A[A_row:] = list[0:50]
        A_row += 1
    
    # 初始化p1和p2
    a = 0.05
    b = 0.01
    p1 =0.5
    p2 =0.5
    Action = 1
    # 在这里使用数字1代表选择动作‘Yes’,用2代表动作‘No’
    for i in range(1):
    
        #         global Action
        # 相似性阈值（the threashhold_value of similarity）
        if (p1 >= p2):
            Action = 1
        else:
            Action = 2
        print('选择的动作是：' + str(Action))
        threshhold_value = 0.3
        similarity = RandomEnviromentForActive(A, n1, n2)
        # p1表示动作1'Yes'被选择的概率，p2表示动作2'No'被选择的概率
        # 前一次选择的动作是‘Yes’，并且该动作得到了奖励
        if (similarity > threshhold_value) and (Action == 1):
            p1 = p1 + a * (1 - p1)
            p2 = 1-p1
           # p2 = (1 - a) * p2
        # 前一次选择的动作是'No',并且该动作得到了奖励
        elif (similarity < threshhold_value) and (Action == 2):
            p2 = (1-a)*p2
            p1 = 1-p2
           # p1 = (1 - a) * p1
        # 前一次选择的动作是‘Yes’，但该动作得到了惩罚
        elif (similarity < threshhold_value) and (Action == 1):
            p2 = 1-b*p2
            p1 = 1-p2
            #p2 = 1 - b * p2
    
        # 前一次选择的动作是‘No’，但该动作得到了惩罚
        elif (similarity > threshhold_value) and (Action == 2):
            p1 = b + (1 - b) * (1 - p1)
            p2 = 1-p1
           # p1 = 1 - b * p1
    
    if (p1 >= p2):
        print('下一时刻选择的动作是:Yes')
    else:
        print('下一时刻选择的动作是:No')
    return p1, p2


#测试主程序
path=r'../Data/itcmatrixs/36000/'
result = np.zeros((50, 50))
for i in os.walk(path):
    for m in range(50):
        for n in range(50):
            r = None
            for j in range(26):
                datapath = path+i[2][j]
                p1,p2 = ContructionAgent(datapath,m,n)
                r = int(p1>=p2)
            result[m,n] = r;
r.save('result.npy') 
pass
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    