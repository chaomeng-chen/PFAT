import seaborn as sns
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import xlrd

# 读取excell的文件
workbook=xlrd.open_workbook('./picExcel.xls')
sheet= workbook.sheet_by_index(0)
X=[]
PDP=[]
PDP_N=[]
FEDAVG=[]
for i in range(1,8):
    X.append(sheet.cell_value(0,i))
    PDP.append(sheet.cell_value(1,i))
    PDP_N.append(sheet.cell_value(2, i))
    FEDAVG.append(sheet.cell_value(3, i))

# 开始画图
# sub_axix = filter(lambda x: x % 200 == 0, x_axix)
plt.title('Result Analysis')
plt.plot(X, PDP, color='green', label='our accuracy')
plt.plot(X, PDP_N, color='red', label='our accuracy without noise')
plt.plot(X, FEDAVG, color='skyblue', label='Fed-Avg accuracy')
# plt.plot(x_axix, thresholds, color='blue', label='threshold')
plt.legend()  # 显示图例
plt.xlabel('number of models')
plt.ylabel('accuracy')
plt.show()

X=[]
PDP=[]
PDP_N=[]
FEDAVG=[]
for i in range(1,8):
    X.append(sheet.cell_value(0,i))
    PDP.append(sheet.cell_value(6,i))
    PDP_N.append(sheet.cell_value(7, i))
    FEDAVG.append(sheet.cell_value(8, i))
plt.title('Result Analysis')
plt.plot(X, PDP, color='green', label='our convergence speed')
plt.plot(X, PDP_N, color='red', label='our convergence speed without noise')
plt.plot(X, FEDAVG, color='skyblue', label='Fed-Avg convergence speed')
# plt.plot(x_axix, thresholds, color='blue', label='threshold')
plt.legend()  # 显示图例
plt.xlabel('number of models')
plt.ylabel('convergence speed')
plt.show()
# 画不同K值下Acc随epoch变化的情况
K=[1,2,5,10,20,40,80]
plt.figure(figsize=(6.4*4, 4.8*2))
for i in range(len(K)):
    plt.subplot(240+i+1)
    X = [j for j in range(1,19)]
    PDP = []
    PDP_N = []
    FEDAVG = []
    for j in range(1, 19):
        PDP.append(sheet.cell_value(14+i, j))
        PDP_N.append(sheet.cell_value(14+i+len(K)+1, j))
        FEDAVG.append(sheet.cell_value(29, j))
    plt.title('K='+str(K[i])+' Result Analysis')
    plt.plot(X, PDP, color='green', label='our accuracy')
    plt.plot(X, PDP_N, color='red', label='our accuracy without noise')
    plt.plot(X, FEDAVG, color='skyblue', label='Fed-Avg accuracy')
    plt.legend()  # 显示图例
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.ylim((0,1))
plt.show()

#画不同K值下Speed随epoch变化的情况
# K=[1,2, 5]
plt.figure(figsize=(6.4*4, 4.8*2))
for i in range(len(K)):
    plt.subplot(240+i+1)
    X = [j for j in range(1,18)]
    PDP = []
    PDP_N = []
    FEDAVG = []
    for j in range(2, 19):
        PDP.append(sheet.cell_value(52+i, j))
        PDP_N.append(sheet.cell_value(52+i+len(K)+1, j))
        FEDAVG.append(sheet.cell_value(67, j))
    plt.title('K='+str(K[i])+' Result Analysis')
    plt.plot(X, PDP, color='green', label='our convergence speed')
    plt.plot(X, PDP_N, color='red', label='our convergence speed without noise')
    plt.plot(X, FEDAVG, color='skyblue', label='Fed-Avg convergence speed')
    # plt.plot(x_axix, thresholds, color='blue', label='threshold')
    plt.legend()  # 显示图例
    plt.xlabel('epochs')
    plt.ylabel('convergence speed')
    plt.ylim((0,40))
plt.show()