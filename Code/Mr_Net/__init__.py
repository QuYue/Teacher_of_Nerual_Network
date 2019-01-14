# -*- coding: utf-8 -*-
"""
@Author: QuYue
@Time  : 2018/12/22 22:22
@File  : __init__.py

Package: Mr Net
A package which can help the neural networks to learn much better.
"""
# %%　Import Packages
import numpy as np
import torch
import matplotlib.pyplot as plt

# %% Help
def help():
    print('Package: Mr Net\nA package which can help the neural networks to learn much better.')

# %% Logistic Regression
class LR_Classifier(torch.nn.Module):
    def __init__(self, feature_dim, label_dim):
        super(LR_Classifier, self).__init__()
        self.linear = torch.nn.Linear(feature_dim, label_dim)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        y = self.sigmoid(x)
        return y


# %% Mini-Batch Gradient descent
class Mr_Result():
    def __init__(self):
        self.train = None
        self.test = None


class Result():
    def __init__(self, correct, total):
        self.correct = correct  #正确
        self.total = total  # 总数
        self.accuracy = correct / total  # 正确率


def compare_label(y, label):
    """
    比较预测结果与真实结果
    :param y: 预测结果
    :param label: 真实结果
    :return: 输出结果（字典：里面有计算完每一个batch后的所有类别的分类结果和整体分类结果（Result类））
    """
    class_amount = y.shape[1]
    pre_y = torch.max(y, 1)[1].data.numpy()  # 得到结果的类别
    label = label.numpy()

    result = dict()
    for i in range(class_amount):
        result[i] = Result(int((pre_y[label == i] == i).sum()), int((label == i).sum()))
    result['Total'] = Result(int((pre_y == label).sum()), len(label))
    return result

def pick_wrong(y, label):
    """
    比较预测结果与真实结果
    :param y: 预测结果
    :param label: 真实结果
    :return: 输出结果（numpy：如果是错误则为1，否则为0）
    """
    class_amount = y.shape[1]
    pre_y = torch.max(y, 1)[1].data.numpy()  # 得到结果的类别
    label = label.numpy()

    wrong = (pre_y != label)
    return wrong

def MGD(network, feature, label, test_feature=None, test_label=None, batch_size=1, lr=0.01, show=True, enable_wrong_repeat = 0):
    """

    :param network: 输入的网络
    :param feature: 特征矩阵（Tensor）
    :param label: 标签（Tensor）
    :param test_feature: 测试集矩阵（Tensor）
    :param test_label: 测试集标签（Tensor）
    :param batch_size: batch的大小（default=1）
    :param lr: 学习率（default=0.01）
    :param show: 是否把每一个batch的结果显示（default=True）
    :return: 输出结果
    """
    N = len(feature)  # data amount
    result = []

    # 测试模式（如果同时test_feature和test_label同时有输入就进入测试模式）
    test_mode = False
    if (type(test_feature) != type(None)) and \
            (type(test_label) != type(None)):
        test_mode = True
    # 训练
    optimizer = torch.optim.SGD(network.parameters(), lr=lr)  #优化器
    criterion = torch.nn.CrossEntropyLoss()  # 损失函数
    for i in range(0, N, batch_size):
        optimizer.zero_grad()
        if enable_wrong_repeat and i:
            wrong_feature = []
            wrong_label = []
            wrong_id = pick_wrong(train_predict_label[i - batch_size: i], label[i - batch_size: i])
            for idx,isWrong in enumerate(wrong_id):
                if isWrong:
                    wrong_feature.append(feature[i - batch_size + idx,:].tolist())
                    wrong_label.append(label[i - batch_size + idx].tolist())
            wrong_feature = torch.Tensor(wrong_feature)
            wrong_label = torch.Tensor(wrong_label).type(torch.LongTensor)

            y = network(torch.cat([feature[i: i + batch_size],wrong_feature],0))  # 预测结果

            loss = criterion(y, torch.cat([label[i: i + batch_size],wrong_label],0))  # 计算损失
        else:
            y = network(feature[i: i + batch_size])  # 预测结果

            loss = criterion(y, label[i: i + batch_size])  #计算损失
        loss.backward()  # 反向传播

        optimizer.step()  # 优化参数

        result_step = Mr_Result()  # 每一步结果用Mr_Result存储

        train_predict_label = network(feature)
        result_step.train = compare_label(train_predict_label, label)  # 训练数据结果

        if test_mode:
            test_predict_label = network(test_feature)
            result_step.test = compare_label(test_predict_label, test_label)  # 测试数据结果
        result.append(result_step)  # 存入结果中

        if show:  # 展示中间过程
            if test_mode:
                print('Train Accuracy is %s, Test Accuracy is %s' % (
                result_step.train['Total'].accuracy, result_step.test['Total'].accuracy))
            else:
                print('Accuracy is %s' % result_step.train['Total'].accuracy)

    return result


# %% Plot
def plot_result(result, lesson_id=''):
    """
    对结果进行画图
    :param result: 结果
    :return: None
    """
    names = list(result[0].train.keys())  # 得到线条的名字
    times = len(result)  # 时间的长度
    # 是否是test_mode
    test_mode = False
    if result[0].test != None:
        test_mode = True

    # 线条
    lines = dict([(i, []) for i in names])  # 训练集线条（字典）
    test_lines = dict([(i, []) for i in names])  # 测试集线条（字典）
    for i in result:
        for j in names:
            lines[j].append(i.train[j].accuracy)  # 添加训练集每一个batch的准确率
            if test_mode:
                test_lines[j].append(i.test[j].accuracy)  # 添加测试集每一个batch的准确率

    plt.figure()
    if test_mode:  # 如果test_mode则画两张图
        plt.subplot(2, 1, 1)
    plots = []
    for i in names:
        if i == 'Total':
            plots.append(plt.plot(list(range(times)), lines[i], linewidth=3)[0])  # 整体
        else:
            plots.append(plt.plot(list(range(times)), lines[i])[0])  # 每个类别
    plt.xlim(0, len(result))  # 图片大小
    plt.ylim(0, 1)
    plt.title('%s Train Accuracy - %s' % (lesson_id, result[-1].train['Total'].accuracy))  # 题目
    plt.legend(plots, names)  # 图例
    plt.grid(True)  # 表格
    if test_mode:
        plt.subplot(2, 1, 2)
        plots = []
        for i in names:
            if i == 'Total':
                plots.append(plt.plot(list(range(times)), test_lines[i], linewidth=3)[0])
            else:
                plots.append(plt.plot(list(range(times)), test_lines[i])[0])
        plt.xlim(0, len(result))
        plt.ylim(0, 1)
        plt.title('%s Test Accuracy - %s' % (lesson_id, result[-1].test['Total'].accuracy))
        plt.legend(plots, names)
        plt.grid(True)
    plt.show()
