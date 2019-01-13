# -*- coding: utf-8 -*-
"""
@Author: QuYue
@Time  : 2019/1/11 12:52
@File  : main_RL.py
"""
# %% Import Packages
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
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
        self.EPOCH = 1 # epoch 迭代次数
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
#%% DQN1
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.8                 # reward discount
TARGET_REPLACE_ITER = 10   # target update frequency
MEMORY_CAPACITY = 2400      # 记忆库容量
N_STATES = 8                # 8个状态：训练和测试的（0，1，2，total准确率）
INPUT_BATCH_SIZE = 10              # batch size
BATCH_SIZE = 800

# Net
class Net(torch.nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(N_STATES, 50) # 特征+label
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.fc2 = torch.nn.Linear(50, 10)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3 = torch.nn.Linear(10, Parm.feature_amount+1)
        self.fc3.weight.data.normal_(0, 0.1)

    def forward(self, x,train_data):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        out = torch.mm(x, train_data.t())
        return out                        # x.data

# DQN
class DQN(object):
    def __init__(self, train_data):
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 1 + INPUT_BATCH_SIZE))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = torch.nn.MSELoss()
        self.train_data = train_data.clone()
        self.train_amount = len(self.train_data)

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() < EPSILON:   # greedy
            actions_value = self.eval_net.forward(x, self.train_data)
            action = actions_value.sort(descending=True)[1][0, :INPUT_BATCH_SIZE].data.numpy()
        else:   # random
            action = np.random.choice(range(self.train_amount), INPUT_BATCH_SIZE)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, r, s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+INPUT_BATCH_SIZE].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+INPUT_BATCH_SIZE:N_STATES+INPUT_BATCH_SIZE+1])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s, self.train_data).gather(1, b_a).mean(1, True) # shape (batch, 1)
        q_next = self.target_net(b_s_, self.train_data).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.sort(1, descending=True)[0][:,:INPUT_BATCH_SIZE].mean(1, True)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# %% Mini-Batch Gradient descent env
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


def MGD(network, feature, label, test_feature=None, test_label=None, batch_size=1, lr=0.01, show=True):
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
        y = network(feature[i: i + batch_size])  # 预测结果

        optimizer.zero_grad()
        loss = criterion(y, label[i: i + batch_size])  #计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 优化参数

        result_step = Mr_Result()  # 每一步结果用Mr_Result存储

        result_step.train = compare_label(network(feature), label)  # 训练数据结果
        if test_mode:
            result_step.test = compare_label(network(test_feature), test_label)  # 测试数据结果
        result.append(result_step)  # 存入结果中

        if show:  # 展示中间过程
            if test_mode:
                print('Train Accuracy is %s, Test Accuracy is %s' % (
                result_step.train['Total'].accuracy, result_step.test['Total'].accuracy))
            else:
                print('Accuracy is %s' % result_step.train['Total'].accuracy)

    return result
#%%
class Environment():
    def __init__(self, Parm, data):
        self.LR = mn.LR_Classifier(Parm.feature_amount, Parm.class_amount)
        self.train_feature = data['train_feature']
        self.train_label = data['train_label']
        self.test_feature = data['test_feature']
        self.test_label = data['test_label']
        self.criterion = torch.nn.CrossEntropyLoss()  # 损失函数
        self.s = None
        self.done = False
        self.counter = 0

    def __get_accuracy(self, feature, label):
        y = self.LR(feature)
        result = compare_label(y, label)
        accuracy = [0] * len(result)  # 状态
        for i in result:
            if i == 'Total':
                accuracy[-1] = result[i].accuracy
            else:
                accuracy[i] = result[i].accuracy
        return accuracy

    def __reward(self, s_):
        reward = 0.
        Total_index = [(int(len(s_) / 2)) - 1, len(s_) - 1]
        if self.done == True:
            reward += ((s_[-1]-0.45)*10)**3

        #fall = (np.array(s_) - np.array(self.s))<0
        #punishment = float(fall.sum() + fall[Total_index].sum()) # total的惩罚为双倍

        r = reward# - punishment
        return r

    def __ifdone(self):
        if self.counter >= 800:
            self.done = True

    def get_state(self):
        s = self.__get_accuracy(self.train_feature, self.train_label) + self.__get_accuracy(self.test_feature, self.test_label)
        return s

    def reset(self, lr=0.01):
        self.LR.load_state_dict(torch.load(Parm.PATH + 'LR1.pth'))
        self.optimizer = torch.optim.SGD(self.LR.parameters(), lr=lr)  # 优化器
        self.s = self.get_state()
        self.done = False
        self.counter = 0
        return self.s

    def step(self, action):
        batch_feature = self.train_feature[action, :]
        batch_label = self.train_label[action]
        y = self.LR(batch_feature)
        loss = self.criterion(y, batch_label)  #计算损失
        self.optimizer.zero_grad()
        loss.backward()  # 反向传播
        self.optimizer.step()  # 优化参数
        s_ = self.get_state()
        self.counter += 1
        self.__ifdone()
        r = self.__reward(s_)
        self.s = s_  # 更新状态
        return s_, r, self.done


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
train_feature, train_label = train_feature0.copy(), train_label0.copy()
# %% To Tensor
train_feature = dataprocess.toTensor(train_feature)
train_label = dataprocess.toTensor(train_label, islabel=True)

test_feature = dataprocess.toTensor(test_feature)
test_label = dataprocess.toTensor(test_label, islabel=True)
data = {'train_feature': train_feature, 'train_label': train_label,
        'test_feature': test_feature, 'test_label': test_label}
train_data = torch.cat((train_feature, train_label.type(torch.FloatTensor).unsqueeze(1)), 1)
# %%
#%%
dqn = DQN(train_data)
env = Environment(Parm, data)
print('\nCollecting experience...')
for i_episode in range(100):
    s = env.reset()
    ep_r = 0
    while True:
        a = dqn.choose_action(s)

        # take action
        s_, r, done = env.step(a)
        dqn.store_transition(s, a, r, s_)

        ep_r += r
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
            if done:
                print('Ep: ', i_episode,
                      '| Ep_r: ', round(ep_r, 2),
                      'Acc:', s_[3], s_[7])
                break

        s = s_
        if done:
            s = env.reset()


