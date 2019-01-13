# -*- coding: utf-8 -*-
"""
@Author: QuYue
@Time  : 2018/12/22 23:34
@File  : main2.py
"""
# %% Import Packages
import pandas as pd
import numpy as np
import torch
import math
from tqdm import tqdm
import Mr_Net as mn
from Mr_Net import dataprocess
from Mr_Net import lesson

# %% Parameter
class Parameter():
    def __init__(self):
        self.PATH = '../Data/'  # path 路径
        self.FILE = 'datingTestSet2.txt'  # file name 文件名
        self.EPOCH = 2 # epoch 迭代次数
        self.TRAINTEST = 4 / 1  # train/test 训练集与测试集的比例

    def stats(self, data):  # statistics 统计
        shape = data.shape
        self.total_amount = shape[0]
        self.train_amount = math.ceil(shape[0] * self.TRAINTEST / (self.TRAINTEST + 1))
        self.test_amount = shape[0] - self.train_amount
        self.feature_amount = shape[1] - 1
        self.class_amount = len(data['label'].unique())

# %% Hyper Parameter
Parm = Parameter()
# %% Read Data
data = pd.read_csv(Parm.PATH + Parm.FILE, sep='\t', names=[
    '飞行公里', '冰淇淋消费', '游戏', 'label'])

Parm.stats(data)  # statistics
# %% Data Processing
feature = data.iloc[:, :Parm.feature_amount]  # feature
label = data.iloc[:, -1]  # label
# feature norm (z-score)
feature = dataprocess.z_score(feature)
# label norm (from 0 to n)
label = dataprocess.zero2n(label)
# split Data
train_feature0, train_label0, test_feature, test_label = mn.dataprocess.split(feature, label, Parm.train_amount, shuffle=True)
# %% Create LR
LR1 = mn.LR_Classifier(Parm.feature_amount, 3)
# torch.save(LR1.state_dict(), Parm.PATH + 'LR1.pth')
# %% Lesson
lessons = ['Lesson1-1', 'Lesson1-2', 'None']
for l in lessons:
    if l == 'Lesson1-1':
        repeat = 1
        new_data = lesson.Lesson1(train_feature0, train_label0, repeat=repeat)
        train_feature, train_label = new_data
    elif l == 'Lesson1-2':
        repeat = 2
        new_data = lesson.Lesson1(train_feature0, train_label0, repeat=repeat)
        train_feature, train_label = new_data
    else:
        repeat = 1
        train_feature, train_label = train_feature0.copy(), train_label0.copy()
    # %% To Tensor
    train_feature = dataprocess.toTensor(train_feature)
    train_label = dataprocess.toTensor(train_label, islabel=True)

    test_feature = dataprocess.toTensor(test_feature)
    test_label = dataprocess.toTensor(test_label, islabel=True)
    # %% Create LR
    LR2 = mn.LR_Classifier(Parm.feature_amount, 3)
    LR2.load_state_dict(torch.load(Parm.PATH + 'LR1.pth'))
    # %% Training
    result = []
    pbar = tqdm(total=round(Parm.EPOCH/repeat))
    print('start')
    for epoch in range(round(Parm.EPOCH/repeat)):
        result += mn.MGD(LR2, train_feature, train_label, test_feature, test_label, batch_size=1, show=False)
        pbar.update(1)
    pbar.close()
    # %% plot
    mn.plot_result(result, l)
