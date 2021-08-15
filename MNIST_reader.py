import os
import struct
import numpy as np
import pickle
from scipy.optimize import minimize
import scipy.stats
import random
import math


"""
Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
which is GPL licensed.
"""

def read(dataset = "training", path = "."):
    if dataset is "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    print(fname_lbl)

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)


    # Reshape and normalize

    img = np.reshape(img, [img.shape[0], img.shape[1]*img.shape[2]])*1.0/255.0

    return img, lbl


def get_data(d):
    # load the data
    x_train, y_train = read('training', d + '/MNIST_original')
    x_test, y_test = read('testing', d + '/MNIST_original')

    # create validation set
    x_vali = list(x_train[50000:].astype(float))
    y_vali = list(y_train[50000:].astype(float))

    # create test_set
    x_train = x_train[:50000].astype(float)
    # 扰动y数据
    y_train = y_train[:50000].astype(float)

    # sort test set (to make federated learning non i.i.d.)
    indices_train = np.argsort(y_train)
    sorted_x_train = list(x_train[indices_train])
    sorted_y_train = list(y_train[indices_train])

    # create a test set
    x_test = list(x_test.astype(float))
    y_test = list(y_test.astype(float))

    return sorted_x_train, sorted_y_train, x_vali, y_vali, x_test, y_test

# idldp
def gen_c_real(max_classes=10,r=[]):
    c_real = np.zeros(max_classes)
    for i in range(500):
        c_real[int(r[i]) - 1] = c_real[int(r[i]) - 1] + 1
    return c_real


def setBudget(minE=1, m=10, l=0):
    budget_list = []
    budget_level_min = []
    budget_level_middle = []
    budget_level_max = []
    every_e = np.zeros(m + l)
    for i in range(m):
        proba = random.random()
        if proba < 0.05:
            every_e[i] = 1
            budget_level_min.append(i + 1)
        elif proba >= 0.05 and proba < 0.1:
            every_e[i] = 2
            budget_level_middle.append(i + 1)
        else:
            every_e[i] = 3
            budget_level_max.append(i + 1)

    # 虚拟项
    for i in range(l):
        every_e[i + m] = 1
        budget_level_min.append(m + i + 1)

    budget_list.append(budget_level_min)
    budget_list.append(budget_level_middle)
    budget_list.append(budget_level_max)

    # print(budget_list)
    m = np.zeros(3)
    m[0] = len(budget_level_min)
    m[1] = len(budget_level_middle)
    m[2] = len(budget_level_max)
    print("every budget num is : ")
    print(m)

    e = np.zeros(3)
    e[0] = minE
    e[1] = 1.2 * minE
    e[2] = 2 * minE
    return e, budget_list, m, every_e

def Perturbation(a=[],b=[],realDataset=[],n=10000,m=100,every_e=[],l=0):
    genData=np.zeros((n,m+l))
    #one-hot编码
    for i in range(n):
        realData=realDataset[i]
        genDatai=np.zeros(m+l)
        genDatai[int(realData)]=1
        genData[i]=genDatai
    #进行扰动
    for i in range(n):
        genDatai=genData[i]
        for j in range(len(genDatai)):
            budget_level = every_e[j]-1
            proba = random.random()
            if (genDatai[j]==0):
                #0转1
                if proba<b[int(budget_level)]:
                    genDatai[j]=1
            else:
                #1转0
                if proba>=a[int(budget_level)]:
                    genDatai[j]=0
        genData[i]=genDatai
    # 计算ci和predict_c
    ci=np.zeros(m)
    for i in range(n):
        for j in range(m):
            if genData[i][j]==1:
                ci[j]=ci[j]+1
    print("gen ci : ")
    print(ci)
    if l==0:
        for i in range(m):
            ci[i]=(ci[i]-n*b[int(every_e[i])-1])/(a[int(every_e[i])-1]-b[int(every_e[i])-1])
    else:
        for i in range(m):
            ci[i]=l*(ci[i]-n*b[int(every_e[i])-1])/(a[int(every_e[i])-1]-b[int(every_e[i])-1])
    #print(ci)
    return genData, ci

# args：m[1..t],x==[τ[1..3]]
def func2(args):
    t=len(args)
    m=args
    #print(m)
    fun = lambda x:(m[0]*(math.e**x[0])/((math.e**x[0])-1)**2)+(m[1]*(math.e**x[1])/((math.e**x[1])-1)**2)+(m[2]*(math.e**x[2])/((math.e**x[2])-1)**2)
    return fun

# args:e[1..3],x[1..3]
def con2(args):
    t=len(args)
    e=args
    #print("bugdet number is : ")
    #print(e)
    cons = ({'type': 'ineq', 'fun': lambda x: x[0]},\
            {'type': 'ineq', 'fun': lambda x: x[1]},\
            {'type': 'ineq', 'fun': lambda x: x[2]},\
            {'type': 'ineq', 'fun': lambda x: min(e[0],e[0]) - (x[0]+x[0])},\
            {'type': 'ineq', 'fun': lambda x: min(e[0],e[1]) - (x[0]+x[1])},\
            {'type': 'ineq', 'fun': lambda x: min(e[0],e[2]) - (x[0]+x[2])},\
            {'type': 'ineq', 'fun': lambda x: min(e[1],e[0]) - (x[1]+x[0])},\
            {'type': 'ineq', 'fun': lambda x: min(e[1],e[1]) - (x[1]+x[1])},\
            {'type': 'ineq', 'fun': lambda x: min(e[1],e[2]) - (x[1]+x[2])},\
            {'type': 'ineq', 'fun': lambda x: min(e[2],e[0]) - (x[2]+x[0])},\
            {'type': 'ineq', 'fun': lambda x: min(e[2],e[1]) - (x[2]+x[1])},\
            {'type': 'ineq', 'fun': lambda x: min(e[2],e[2]) - (x[2]+x[2])},
            )
    return cons

def getBestPerturbation(budget=[],m=[]):
    args = m
    args1 = budget
    cons = con2(args1)
    x0 = np.ones(len(m))
    res = minimize(func2(args), x0, method='SLSQP', constraints=cons)
    a = np.zeros(len(m))
    b = np.zeros(len(m))
    for i in range(len(m)):
        a[i] = (math.e ** res.x[i]) / ((math.e ** res.x[i]) + 1)
        b[i] = 1 / ((math.e ** res.x[i]) + 1)
    return res.success, a, b

class Data:
    def __init__(self, save_dir, n):
        raw_directory = save_dir + '/DATA'
        self.client_set = pickle.load(open(raw_directory + '/clients/' + str(n) + '_clients.pkl', 'rb'))
        self.sorted_x_train, self.sorted_y_train, self.x_vali, self.y_vali, self.x_test, self.y_test = get_data(save_dir)

        # 1.0
        # 扰动y数据
        # label_set_asarray = np.asarray(self.sorted_y_train)
        # for i in range(n):
        #     batch_ind = self.client_set[i]
        #     # 一台客户机上的所有y标签
        #     realDataset = label_set_asarray[[int(j) for j in batch_ind]]
        #     # y标签的样本情况
        #     y_val = np.unique(realDataset)
        #     # 计算原始每个标签对应的样本数c_real
        #     c_real = gen_c_real(max_classes=10, r=realDataset)
        #     # 为每个类别分配隐私预算，值为budget_val，统计每个隐私预算对应的类别every_e，每个隐私预算的类别数量m
        #     budget_val, _,  m, every_e = setBudget(minE=1, m=10)
        #     # 得到最优扰动参数a,b
        #     result, a, b = getBestPerturbation(budget=budget_val, m=m)
        #     if result == True:
        #         # 进行扰动
        #         genData, c_predict = Perturbation(a=a, b=b, realDataset=realDataset, n=500, m=10, every_e=every_e)
        #         for j in range(500):
        #             choice_list = []
        #             if genData[j][int(realDataset[j])]==1:
        #                 choose=realDataset[j]
        #             else:
        #                 for k in range(10):
        #                     if genData[j][k]>0:
        #                         choice_list.append(k)
        #                 if len(choice_list)>0:
        #                     choose = np.random.choice(choice_list)
        #                 else:
        #                     choose = np.random.choice(10)
        #             # if realDataset[j]==y_val[0]:
        #             #     if genData[j][y_val[0]]==1:
        #             #         choose=y_val[0]
        #             #     else:
        #             #         choose=y_val[1]
        #             # else:
        #             #     if genData[j][y_val[1]]==1:
        #             #         choose=y_val[1]
        #             #     else:
        #             #         choose=y_val[0]
        #             self.sorted_y_train[int(batch_ind[j])]=choose
        #end 1.0