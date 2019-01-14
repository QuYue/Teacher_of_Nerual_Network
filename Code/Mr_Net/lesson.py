# -*- coding: utf-8 -*-
"""
@Author: QuYue
@Time  : 2018/12/24 15:18
@File  : lesson.py

Package: Mr Net.lesson
A package which can help the neural networks to learn much better.
"""
#%% Import Packages
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import torch

#%% Lesson 1
# ---------- #
# I assume if the feature' s value which be recomputed after normalization is more close to the zero, the information included by this feature is lower.
# So the simple questions are the samples which the sum of values are low.
# infor_level = sum(abs(values))
# ---------- #
def Lesson1(feature, label, repeat = 1):
    # 信息量
    def infor_level(record):
        return record.abs().sum() - np.abs(record['label'])
    # 重复次数
    def repeat_record(record):
        index = []
        #N = record.shape[0] # sample amount
        for i in range(len(record)):
            index.extend([i] * repeat)
        new_record = record.iloc[index[:]]
        return new_record

    #N = len(feature)

    data0 = pd.DataFrame(feature.copy())
    data0['label'] = label

    data0['infor_level'] = data0.apply(infor_level, axis=1)
    data0 = data0.sort_values(by='infor_level',ascending=True)

    new_data = data0.apply(repeat_record)
    feature = new_data.iloc[:, :-2]
    label = new_data['label']

    return feature, label
#%% Lesson 2
def Lesson2(feature, label, repeat = 1, rate = 0.2):
    # 信息量
    def infor_level(record):
        # info = (record.abs()<rate).sum() - (np.abs(record['label'])<rate)
        # print(info)
        return (record.abs()<rate).sum() - (np.abs(record['label'])<rate)
    # 重复次数
    def repeat_record(record):
        index = []
        #N = record.shape[0] # sample amount
        for i in range(len(record)):
            index.extend([i]*repeat)
        new_record = record.iloc[index[:]]
        return new_record

    #N = len(feature)  # sample amount

    data0 = pd.DataFrame(feature.copy())
    data0['label'] = label

    data0['infor_level'] = data0.apply(infor_level, axis=1)
    data0 = data0.sort_values(by='infor_level',ascending=True)

    new_data = data0.apply(repeat_record)
    feature = new_data.iloc[:, :-2]
    label = new_data['label']

    return feature, label

#%% Lesson kmean
def Lesson_kmean(feature, label, repeat = 1, n_cluster = 3):
    # 信息量
    def kmean(record):
        # info = (record.abs()<rate).sum() - (np.abs(record['label'])<rate)
        # print(info)
        clf = KMeans(n_clusters=n_cluster, max_iter=300, n_init=10)
        return clf.fit_predict(record)
    # 重复次数
    def repeat_record(record):
        index = []
        #N = record.shape[0] # sample amount
        for i in range(len(record)):
            index.extend([i]*repeat)
        new_record = record.iloc[index[:]]
        return new_record

    #N = len(feature)  # sample amount
    cluster =  kmean(feature)

    data0 = pd.DataFrame(feature.copy())

    data0['cluster'] = cluster
    data0['label'] = label
    data0 = data0.sort_values(by='cluster',ascending=True)

    new_data = data0.apply(repeat_record)
    feature = new_data.iloc[:, :-2]
    label = new_data['label']

    return feature, label

#%% Lesson DBSCAN
def Lesson_DBSCAN(feature, label, repeat = 1, n_cluster = 3):
    # 信息量
    def dbscan(record):
        # info = (record.abs()<rate).sum() - (np.abs(record['label'])<rate)
        # print(info)
        clf = DBSCAN(eps=0.1,  # 邻域半径

                min_samples=1,    # 最小样本点数，MinPts

                metric='hamming', #'manhattan','euclidean','hamming',‘jaccard’

                metric_params=None,

                algorithm='auto', # 'auto','ball_tree','kd_tree','brute',4个可选的参数 寻找最近邻点的算法，例如直接密度可达的点

                leaf_size=30, # balltree,cdtree的参数

                p=None, #

                n_jobs=1)

        return clf.fit_predict(record)
    # 重复次数
    def repeat_record(record):
        index = []
        #N = record.shape[0] # sample amount
        for i in range(len(record)):
            index.extend([i]*repeat)
        new_record = record.iloc[index[:]]
        return new_record

    #N = len(feature)  # sample amount
    cluster =  dbscan(feature)

    data0 = pd.DataFrame(feature.copy())

    data0['cluster'] = cluster
    data0['label'] = label
    data0 = data0.sort_values(by='cluster',ascending=True)

    new_data = data0.apply(repeat_record)
    feature = new_data.iloc[:, :-2]
    label = new_data['label']

    return feature, label
'''
#%% Lesson RL_batch_input
def Lesson_DBSCAN(feature, label, repeat = 1, n_cluster = 3):
    # 信息量
    def dbscan(record):
        # info = (record.abs()<rate).sum() - (np.abs(record['label'])<rate)
        # print(info)
        clf = DBSCAN(eps=0.1,  # 邻域半径

                min_samples=1,    # 最小样本点数，MinPts

                metric='euclidean', #'manhattan','euclidean','hamming',‘jaccard’

                metric_params=None,

                algorithm='auto', # 'auto','ball_tree','kd_tree','brute',4个可选的参数 寻找最近邻点的算法，例如直接密度可达的点

                leaf_size=30, # balltree,cdtree的参数

                p=None, #

                n_jobs=1)

        return clf.fit_predict(record)
    # 重复次数
    def repeat_record(record):
        index = []
        #N = record.shape[0] # sample amount
        for i in range(len(record)):
            index.extend([i]*repeat)
        new_record = record.iloc[index[:]]
        return new_record

    #N = len(feature)  # sample amount
    cluster =  dbscan(feature)

    data0 = pd.DataFrame(feature.copy())

    data0['cluster'] = cluster
    data0['label'] = label
    data0 = data0.sort_values(by='cluster',ascending=True)

    new_data = data0.apply(repeat_record)
    feature = new_data.iloc[:, :-2]
    label = new_data['label']

    return feature, label
'''
