#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import scipy.sparse as sp 

import torch
from torch.utils.data import Dataset, DataLoader

# 打印矩阵的统计信息，如平均交互数、非零行和列的比例、矩阵密度等
def print_statistics(X, string):
    print('>'*10 + string + '>'*10 )
    # 计算矩阵每行交互数的平均值
    print('Average interactions', X.sum(1).mean(0).item())
    # 获取矩阵中非零元素的行索引和列索引
    nonzero_row_indice, nonzero_col_indice = X.nonzero()
    # 获取非零行的唯一索引
    unique_nonzero_row_indice = np.unique(nonzero_row_indice)
    # 获取非零列的唯一索引
    unique_nonzero_col_indice = np.unique(nonzero_col_indice)
    # 计算非零行的比例
    print('Non-zero rows', len(unique_nonzero_row_indice)/X.shape[0])
    # 计算非零列的比例
    print('Non-zero columns', len(unique_nonzero_col_indice)/X.shape[1])
    # 计算矩阵的密度
    print('Matrix density', len(nonzero_row_indice)/(X.shape[0]*X.shape[1]))

# 自定义数据集类，用于训练阶段的捆绑包数据
class BundleTrainDataset(Dataset):
    def __init__(self, conf, u_b_pairs, u_b_graph, num_bundles, u_b_for_neg_sample, b_b_for_neg_sample, neg_sample=1):
        # 保存配置信息
        self.conf = conf
        # 用户 - 捆绑包正样本对
        self.u_b_pairs = u_b_pairs
        # 用户 - 捆绑包交互图（稀疏矩阵）
        self.u_b_graph = u_b_graph
        # 捆绑包的总数
        self.num_bundles = num_bundles
        # 负样本的数量
        self.neg_sample = neg_sample

        # 用于负采样的用户 - 捆绑包信息
        self.u_b_for_neg_sample = u_b_for_neg_sample
        # 用于负采样的捆绑包 - 捆绑包信息
        self.b_b_for_neg_sample = b_b_for_neg_sample

        # 计算每个捆绑包的交互量
        self.bundle_interaction_counts = self.calculate_bundle_interaction_counts()
        # 筛选高交互量的捆绑包
        self.high_interaction_bundles = [bundle for bundle, count in self.bundle_interaction_counts.items() if count > self.conf["interaction_threshold"]]

    def calculate_bundle_interaction_counts(self):
        bundle_interaction_counts = {}
        for user, bundle in self.u_b_pairs:
            if bundle in bundle_interaction_counts:
                bundle_interaction_counts[bundle] += 1
            else:
                bundle_interaction_counts[bundle] = 1
        return bundle_interaction_counts

    # 根据索引获取一个样本
    def __getitem__(self, index):
        conf = self.conf
        # 获取当前索引对应的用户和正样本捆绑包
        user_b, pos_bundle = self.u_b_pairs[index]
        # 初始化一个列表，用于存储正样本和负样本捆绑包
        all_bundles = [pos_bundle]

        # 进行负采样
        user_interacted_bundles = set(self.u_b_graph[user_b].nonzero()[1])
        possible_neg_bundles = [bundle for bundle in self.high_interaction_bundles if bundle not in user_interacted_bundles]

        if len(possible_neg_bundles) < self.neg_sample:
            # 如果可用的负样本不足，从所有未交互的捆绑包中采样
            all_non_interacted_bundles = [bundle for bundle in range(self.num_bundles) if bundle not in user_interacted_bundles]
            possible_neg_bundles = all_non_interacted_bundles

        neg_bundles = np.random.choice(possible_neg_bundles, self.neg_sample, replace=False)
        all_bundles.extend(neg_bundles)

        # 将用户索引和捆绑包索引转换为 PyTorch 的 LongTensor 类型并返回
        return torch.LongTensor([user_b]), torch.LongTensor(all_bundles)

    # 返回数据集的长度
    def __len__(self):
        return len(self.u_b_pairs)

# 自定义数据集类，用于测试阶段的捆绑包数据
class BundleTestDataset(Dataset):
    def __init__(self, u_b_pairs, u_b_graph, u_b_graph_train, num_users, num_bundles):
        # 用户 - 捆绑包对
        self.u_b_pairs = u_b_pairs
        # 用户 - 捆绑包交互图（测试集）
        self.u_b_graph = u_b_graph
        # 用户 - 捆绑包交互图（训练集），用于屏蔽训练集中已有的交互
        self.train_mask_u_b = u_b_graph_train

        # 用户的总数
        self.num_users = num_users
        # 捆绑包的总数
        self.num_bundles = num_bundles

        # 生成所有用户的索引张量
        self.users = torch.arange(num_users, dtype=torch.long).unsqueeze(dim=1)
        # 生成所有捆绑包的索引张量
        self.bundles = torch.arange(num_bundles, dtype=torch.long)

    # 根据索引获取一个样本
    def __getitem__(self, index):
        # 将测试集中指定用户的交互图转换为 PyTorch 张量
        u_b_grd = torch.from_numpy(self.u_b_graph[index].toarray()).squeeze()
        # 将训练集中指定用户的交互图转换为 PyTorch 张量
        u_b_mask = torch.from_numpy(self.train_mask_u_b[index].toarray()).squeeze()

        return index, u_b_grd, u_b_mask

    # 返回数据集的长度
    def __len__(self):
        return self.u_b_graph.shape[0]

# 数据集管理类，用于加载和处理所有数据
class Datasets():
    def __init__(self, conf):
        # 数据文件的路径
        self.path = conf['data_path']
        # 数据集的名称
        self.name = conf['dataset']
        # 训练集的批量大小
        batch_size_train = conf['batch_size_train']
        # 测试集和验证集的批量大小
        batch_size_test = conf['batch_size_test']

        # 获取数据的基本信息，如用户数、捆绑包数和物品数
        self.num_users, self.num_bundles, self.num_items = self.get_data_size()

        # 获取捆绑包 - 物品的交互对和交互图
        b_i_pairs, b_i_graph = self.get_bi()
        # 获取用户 - 物品的交互对和交互图
        u_i_pairs, u_i_graph = self.get_ui() #u_i_pairs 似乎没用上
        # 获取物品 - 物品的交互图
        i_i_graph = self.get_ii()

        # 获取训练集的用户 - 捆绑包交互对和交互图
        u_b_pairs_train, u_b_graph_train = self.get_ub("train")
        # 获取验证集的用户 - 捆绑包交互对和交互图
        u_b_pairs_val, u_b_graph_val = self.get_ub("tune")
        # 获取测试集的用户 - 捆绑包交互对和交互图
        u_b_pairs_test, u_b_graph_test = self.get_ub("test")

        # 用于负采样的信息，这里暂时设为 None
        u_b_for_neg_sample, b_b_for_neg_sample = None, None

        # 创建训练集的数据集对象
        self.bundle_train_data = BundleTrainDataset(conf, u_b_pairs_train, u_b_graph_train, self.num_bundles, u_b_for_neg_sample, b_b_for_neg_sample, conf["neg_num"])
        # 创建验证集的数据集对象
        self.bundle_val_data = BundleTestDataset(u_b_pairs_val, u_b_graph_val, u_b_graph_train, self.num_users, self.num_bundles)
        # 创建测试集的数据集对象
        self.bundle_test_data = BundleTestDataset(u_b_pairs_test, u_b_graph_test, u_b_graph_train, self.num_users, self.num_bundles)

        # 存储用户 - 捆绑包、用户 - 物品、捆绑包 - 物品的交互图
        # self.graphs = [u_b_graph_train, u_i_graph, b_i_graph]
        self.graphs = [u_b_graph_train, u_i_graph, b_i_graph, i_i_graph]

        # 创建训练集的数据加载器
        self.train_loader = DataLoader(self.bundle_train_data, batch_size=batch_size_train, shuffle=True, num_workers=10, drop_last=True)
        # 创建验证集的数据加载器
        self.val_loader = DataLoader(self.bundle_val_data, batch_size=batch_size_test, shuffle=False, num_workers=20)
        # 创建测试集的数据加载器
        self.test_loader = DataLoader(self.bundle_test_data, batch_size=batch_size_test, shuffle=False, num_workers=20)
    #     # bundle出现次数计数器
    #     self.bundle_interaction_counts = self.calculate_bundle_interaction_counts()
    #     # 筛选高交互量的捆绑包
    #     self.high_interaction_bundles = [bundle for bundle, count in self.bundle_interaction_counts.items() if count > self.conf["interaction_threshold"]]

    # def get_neg_samples(self, user):
    #     # 获取与当前用户交互的捆绑包
    #     user_interacted_bundles = set(self.u_b_graph[user].nonzero()[1])
    #     # 从高交互量且未与当前用户交互的捆绑包中采样负样本
    #     possible_neg_bundles = [bundle for bundle in self.high_interaction_bundles if bundle not in user_interacted_bundles]
    #     if len(possible_neg_bundles) < self.conf["neg_num"]:
    #         # 如果可用的负样本不足，从所有未交互的捆绑包中采样
    #         all_non_interacted_bundles = [bundle for bundle in range(self.num_bundles) if bundle not in user_interacted_bundles]
    #         possible_neg_bundles = all_non_interacted_bundles
    #     neg_bundles = np.random.choice(possible_neg_bundles, self.conf["neg_num"], replace=False)
    #     return neg_bundles

    # def calculate_bundle_interaction_counts(self):
    #     # 假设u_b_pairs是用户 - 捆绑包正样本对
    #     u_b_pairs = self.get_ub('train')[0]
    #     bundle_interaction_counts = {}
    #     for _, bundle in u_b_pairs:
    #         if bundle in bundle_interaction_counts:
    #             bundle_interaction_counts[bundle] += 1
    #         else:
    #             bundle_interaction_counts[bundle] = 1
    #     return bundle_interaction_counts

    # 获取数据的基本信息，如用户数、捆绑包数和物品数
    def get_data_size(self):
        name = self.name
        # 如果数据集名称包含下划线，取下划线前的部分
        if "_" in name:
            name = name.split("_")[0]
        # 打开数据大小信息文件
        with open(os.path.join(self.path, self.name, '{}_data_size.txt'.format(name)), 'r') as f:
            # 读取文件中的数据大小信息，并转换为整数列表，取前三个值
            return [int(s) for s in f.readline().split('\t')][:3]

    # 获取捆绑包 - 物品的交互对和交互图
    def get_bi(self):
        # 打开捆绑包 - 物品交互信息文件
        with open(os.path.join(self.path, self.name, 'bundle_item.txt'), 'r') as f:
            # 读取文件中的交互对信息，并转换为元组列表
            b_i_pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))

        # 将交互对信息转换为 numpy 数组
        indice = np.array(b_i_pairs, dtype=np.int32)
        # 创建一个全为 1 的值数组，表示交互存在
        values = np.ones(len(b_i_pairs), dtype=np.float32)
        # 创建稀疏矩阵表示捆绑包 - 物品的交互图
        b_i_graph = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_bundles, self.num_items)).tocsr()

        # 打印捆绑包 - 物品交互图的统计信息
        print_statistics(b_i_graph, 'B-I statistics')

        return b_i_pairs, b_i_graph

    # 获取用户 - 物品的交互对和交互图
    def get_ui(self):
        # 打开用户 - 物品交互信息文件
        with open(os.path.join(self.path, self.name, 'user_item.txt'), 'r') as f:
            # 读取文件中的交互对信息，并转换为元组列表
            u_i_pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))

        # 将交互对信息转换为 numpy 数组
        indice = np.array(u_i_pairs, dtype=np.int32)
        # 创建一个全为 1 的值数组，表示交互存在
        values = np.ones(len(u_i_pairs), dtype=np.float32)
        # 创建稀疏矩阵表示用户 - 物品的交互图
        u_i_graph = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_users, self.num_items)).tocsr()

        # 打印用户 - 物品交互图的统计信息
        print_statistics(u_i_graph, 'U-I statistics')

        return u_i_pairs, u_i_graph

    # 获取用户 - 捆绑包的交互对和交互图，根据不同的任务（训练、验证、测试）
    def get_ub(self, task):
        # 打开指定任务的用户 - 捆绑包交互信息文件
        with open(os.path.join(self.path, self.name, 'user_bundle_{}.txt'.format(task)), 'r') as f:
            # 读取文件中的交互对信息，并转换为元组列表
            u_b_pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))

        # 将交互对信息转换为 numpy 数组
        indice = np.array(u_b_pairs, dtype=np.int32)
        # 创建一个全为 1 的值数组，表示交互存在
        values = np.ones(len(u_b_pairs), dtype=np.float32)
        # 创建稀疏矩阵表示用户 - 捆绑包的交互图
        u_b_graph = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_users, self.num_bundles)).tocsr()

        # 打印指定任务的用户 - 捆绑包交互图的统计信息
        print_statistics(u_b_graph, "U-B statistics in %s" %(task))

        return u_b_pairs, u_b_graph


    def get_ii(self):
        # 打开物品 - 物品交互信息文件
        with open(os.path.join(self.path, self.name, 'II_matrix_all.txt'), 'r') as f:
            lines = f.readlines()
            # 读取文件中的交互对信息，并转换为元组列表
            i_i_pairs = [tuple(int(i) for i in line[:-1].split(' ')[:2]) for line in lines]
            values = np.array([float(line[:-1].split(' ')[2]) for line in lines])
        para = [0.6,0.8,1]
        # 使用 numpy 条件赋值修改值
        values = np.where(values == 1, para[0], values)#bi高
        values = np.where(values == 2, para[1], values)#ui高bi低
        values = np.where(values == 3, para[2], values)

        # 将交互对信息转换为 numpy 数组
        indice = np.array(i_i_pairs, dtype=np.int32)
        # 创建稀疏矩阵表示物品 - 物品的交互图
        i_i_graph = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_items, self.num_items)).tocsr()
        # 打印物品 - 物品交互图的统计信息
        # print_statistics(i_i_graph, 'I-I statistics')
        print('\n'+'*'*10 + " para: " + str(para) + ' ' + '*'*10+'\n')
        return i_i_graph
    
## 0  ,0.8,1 ui@bi at lap on both  0.03351
## 0.2,0.8,1 ui@bi at lap on both  2025-03-19 20:56:37, Best in epoch 9, TOP 20: REC_T=0.03430, NDCG_T=0.01860
## 0.4,0.8,1 ui&bi at lap on both  0.03363