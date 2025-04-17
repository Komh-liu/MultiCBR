#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp 
from scipy.sparse import csr_matrix, vstack , diags

# 计算 BPR（Bayesian Personalized Ranking）损失
def cal_bpr_loss(pred):
    # pred: [bs, 1+neg_num]，bs 表示批量大小，neg_num 表示负样本数量
    if pred.shape[1] > 2:
        # 提取负样本的预测分数
        negs = pred[:, 1:]
        # 提取正样本的预测分数，并扩展维度以与负样本匹配
        pos = pred[:, 0].unsqueeze(1).expand_as(negs)
    else:
        # 当负样本数量为 1 时的处理
        negs = pred[:, 1].unsqueeze(1)
        pos = pred[:, 0].unsqueeze(1)

    # 计算 BPR 损失，使用 sigmoid 函数和对数函数
    loss = - torch.log(torch.sigmoid(pos - negs))  # [bs]
    # 计算平均损失
    loss = torch.mean(loss)

    return loss

# 对图进行拉普拉斯变换
def laplace_transform(graph):
    # 计算图的每行元素之和的平方根的倒数，构建对角矩阵
    rowsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=1).A.ravel()) + 1e-8))
    # 计算图的每列元素之和的平方根的倒数，构建对角矩阵
    colsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=0).A.ravel()) + 1e-8))
    # 进行拉普拉斯变换
    graph = rowsum_sqrt @ graph @ colsum_sqrt

    return graph

# 将稀疏矩阵转换为 PyTorch 稀疏张量
def to_tensor(graph):
    # 将图转换为 COO 格式的稀疏矩阵
    graph = graph.tocoo()
    # 提取图的非零元素值
    values = graph.data
    # 提取非零元素的行索引和列索引
    indices = np.vstack((graph.row, graph.col))
    # 创建 PyTorch 稀疏浮点张量
    graph = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(graph.shape))

    return graph

# 对图的边进行随机丢弃
def np_edge_dropout(values, dropout_ratio):
    # 以 dropout_ratio 的概率生成 0 或 1 的掩码
    mask = np.random.choice([0, 1], size=(len(values),), p=[dropout_ratio, 1-dropout_ratio])
    # 根据掩码丢弃边
    values = mask * values
    return values

# 定义 MultiCBR 模型类
class MultiCBR(nn.Module):
    def __init__(self, conf, raw_graph):
        # 调用父类的构造函数
        super().__init__()
        # 保存配置信息
        self.conf = conf
        # 获取设备信息，如 CPU 或 GPU
        device = self.conf["device"]
        self.device = device

        # 嵌入向量的维度
        self.embedding_size = conf["embedding_size"]
        # L2 正则化系数
        self.embed_L2_norm = conf["l2_reg"]
        # 用户数量
        self.num_users = conf["num_users"]
        # 捆绑包数量
        self.num_bundles = conf["num_bundles"]
        # 物品数量
        self.num_items = conf["num_items"]
        # 传播层数
        self.num_layers = self.conf["num_layers"]
        # 对比损失的温度参数
        self.c_temp = self.conf["c_temp"]

        # 融合权重配置
        self.fusion_weights = conf['fusion_weights']

        # 初始化嵌入向量
        self.init_emb()
        # 初始化融合权重
        self.init_fusion_weights()

        # 确保原始图是列表类型
        assert isinstance(raw_graph, list)
        # 提取用户 - 捆绑包图、用户 - 物品图和捆绑包 - 物品图
        # self.ub_graph, self.ui_graph, self.bi_graph = raw_graph
        # 提取用户 - 捆绑包图、用户 - 物品图、捆绑包 - 物品图和物品 - 物品图
        self.ub_graph, self.ui_graph, self.bi_graph, self.ii_graph, self.w_bi_graph = raw_graph

        self.II_propagation_graph = to_tensor(laplace_transform(self.ii_graph)).to(device)

        # 生成用于测试的无丢弃的传播图
        self.UB_propagation_graph_ori = self.get_propagation_graph(self.ub_graph)

        #注意 修改所有图时要注意同时修改训练用图和测试用图！！！
        #注意 修改所有图时要注意同时修改训练用图和测试用图！！！
        #注意 修改所有图时要注意同时修改训练用图和测试用图！！！
        
        # self.UI_propagation_graph_ori = self.get_propagation_graph_with_ii(self.ui_graph, self.ii_graph)
        # self.UI_propagation_graph_ori = self.get_propagation_graph(self.multiply_and_normalize(self.ui_graph,self.ii_graph))
        # self.UI_aggregation_graph_ori = self.get_aggregation_graph(self.multiply_and_normalize(self.ui_graph,self.ii_graph))
        self.UI_propagation_graph_ori = self.get_propagation_graph(self.ui_graph)
        self.UI_aggregation_graph_ori = self.get_aggregation_graph(self.ui_graph)

        # self.BI_propagation_graph_ori = self.get_propagation_graph(laplace_transform(self.bi_graph@self.ii_graph))
        # self.BI_aggregation_graph_ori = self.get_aggregation_graph(laplace_transform(self.bi_graph@self.ii_graph))
        self.BI_propagation_graph_ori = self.get_propagation_graph(self.bi_graph)
        self.BI_aggregation_graph_ori = self.get_aggregation_graph(self.bi_graph)

        # 生成用于训练的带有配置丢弃率的传播图
        # 如果增强类型是 OP 或 MD，这些图将与上面的相同
        self.UB_propagation_graph = self.get_propagation_graph(self.ub_graph, self.conf["UB_ratio"])

        # self.UI_propagation_graph = self.get_propagation_graph_with_ii(self.ui_graph, self.ii_graph, self.conf["UI_ratio"])
        self.UI_propagation_graph = self.get_propagation_graph(self.ui_graph, self.conf["UI_ratio"])
        self.UI_aggregation_graph = self.get_aggregation_graph(self.ui_graph, self.conf["UI_ratio"])
        # self.UI_propagation_graph = self.get_propagation_graph(self.multiply_and_normalize(self.ui_graph,self.ii_graph), self.conf["UI_ratio"])
        # self.UI_aggregation_graph = self.get_aggregation_graph(self.multiply_and_normalize(self.ui_graph,self.ii_graph), self.conf["UI_ratio"])

        # self.BI_propagation_graph = self.get_propagation_graph_with_ii(self.bi_graph, self.ii_graph, self.conf["BI_ratio"])
        self.BI_propagation_graph = self.get_propagation_graph(self.bi_graph, self.conf["BI_ratio"])
        self.BI_aggregation_graph = self.get_aggregation_graph(self.bi_graph, self.conf["BI_ratio"])

        # self.BI_propagation_graph = self.get_propagation_graph(laplace_transform(self.bi_graph@self.ii_graph), self.conf["BI_ratio"])
        # self.BI_aggregation_graph = self.get_aggregation_graph(laplace_transform(self.bi_graph@self.ii_graph), self.conf["BI_ratio"])

        self.UB_aggregation_graph = self.get_aggregation_graph(self.ub_graph, self.conf["UB_ratio"])
        self.BU_aggregation_graph = torch.transpose(self.UB_aggregation_graph, 0, 1) ## 需要转置
        
        # 如果增强类型是 MD，初始化模态丢弃层
        if self.conf['aug_type'] == 'MD':
            self.init_md_dropouts()
        # 如果增强类型是 Noise，初始化噪声参数
        elif self.conf['aug_type'] == "Noise":
            self.init_noise_eps()

        self.item_relations = self.load_item_relations()
        
    # 初始化模态丢弃层
    def init_md_dropouts(self):
        # 初始化用户 - 捆绑包图的丢弃层
        self.UB_dropout = nn.Dropout(self.conf["UB_ratio"], True)
        # 初始化用户 - 物品图的丢弃层
        self.UI_dropout = nn.Dropout(self.conf["UI_ratio"], True)
        # 初始化捆绑包 - 物品图的丢弃层
        self.BI_dropout = nn.Dropout(self.conf["BI_ratio"], True)
        # 存储丢弃层的字典
        self.mess_dropout_dict = {
            "UB": self.UB_dropout,
            "UI": self.UI_dropout,
            "BI": self.BI_dropout
        }

    # 初始化噪声参数
    def init_noise_eps(self):
        # 用户 - 捆绑包图的噪声参数
        self.UB_eps = self.conf["UB_ratio"]
        # 用户 - 物品图的噪声参数
        self.UI_eps = self.conf["UI_ratio"]
        # 捆绑包 - 物品图的噪声参数
        self.BI_eps = self.conf["BI_ratio"]
        # 存储噪声参数的字典
        self.eps_dict = {
            "UB": self.UB_eps,
            "UI": self.UI_eps,
            "BI": self.BI_eps
        }

    # 初始化嵌入向量
    def init_emb(self):
        # 初始化用户嵌入向量
        self.users_feature = nn.Parameter(torch.FloatTensor(self.num_users, self.embedding_size))
        # 使用 Xavier 正态分布初始化用户嵌入向量
        nn.init.xavier_normal_(self.users_feature)
        # 初始化捆绑包嵌入向量
        self.bundles_feature = nn.Parameter(torch.FloatTensor(self.num_bundles, self.embedding_size))
        # 使用 Xavier 正态分布初始化捆绑包嵌入向量
        nn.init.xavier_normal_(self.bundles_feature)
        # 初始化物品嵌入向量
        self.items_feature = nn.Parameter(torch.FloatTensor(self.num_items, self.embedding_size))
        # 使用 Xavier 正态分布初始化物品嵌入向量
        nn.init.xavier_normal_(self.items_feature)

    # 初始化融合权重
    def init_fusion_weights(self):
        # 确保模态融合权重的数量与图的数量一致
        assert (len(self.fusion_weights['modal_weight']) == 3), \
            "The number of modal fusion weights does not correspond to the number of graphs"

        # 确保层融合权重的数量与层数一致
        assert (len(self.fusion_weights['UB_layer']) == self.num_layers + 1) and\
               (len(self.fusion_weights['UI_layer']) == self.num_layers + 1) and \
               (len(self.fusion_weights['BI_layer']) == self.num_layers + 1) and \
                (len(self.fusion_weights['II_layer']) == self.num_layers + 1), \
            "The number of layer fusion weights does not correspond to number of layers"

        # 将模态融合权重转换为 PyTorch 张量
        modal_coefs = torch.FloatTensor(self.fusion_weights['modal_weight'])
        # 将用户 - 捆绑包图层融合权重转换为 PyTorch 张量
        UB_layer_coefs = torch.FloatTensor(self.fusion_weights['UB_layer'])
        # 将用户 - 物品图层融合权重转换为 PyTorch 张量
        UI_layer_coefs = torch.FloatTensor(self.fusion_weights['UI_layer'])
        # 将捆绑包 - 物品图层融合权重转换为 PyTorch 张量
        BI_layer_coefs = torch.FloatTensor(self.fusion_weights['BI_layer'])
        # 将物品 - 物品图层融合权重转换为 PyTorch 张量
        II_layer_coefs = torch.FloatTensor(self.fusion_weights['II_layer'])

        # 扩展模态融合权重的维度并移动到指定设备
        self.modal_coefs = modal_coefs.unsqueeze(-1).unsqueeze(-1).to(self.device)

        # 扩展用户 - 捆绑包图层融合权重的维度并移动到指定设备
        self.UB_layer_coefs = UB_layer_coefs.unsqueeze(0).unsqueeze(-1).to(self.device)
        # 扩展用户 - 物品图层融合权重的维度并移动到指定设备
        self.UI_layer_coefs = UI_layer_coefs.unsqueeze(0).unsqueeze(-1).to(self.device)
        # 扩展捆绑包 - 物品图层融合权重的维度并移动到指定设备
        self.BI_layer_coefs = BI_layer_coefs.unsqueeze(0).unsqueeze(-1).to(self.device)
        # 扩展物品 - 物品图层融合权重的维度并移动到指定设备
        self.II_layer_coefs = II_layer_coefs.unsqueeze(0).unsqueeze(-1).to(self.device)

    # def multiply_and_normalize(self, x_i_graph, ii_graph):
    #     # 进行矩阵乘法
    #     multiplied_graph = x_i_graph @ ii_graph

    #     # 将矩阵中所有值置为 1
    #     multiplied_graph.data = np.ones_like(multiplied_graph.data)

    #     # 进行 Laplace 归一化
    #     normalized_graph = laplace_transform(multiplied_graph)

    #     return normalized_graph

    def multiply_and_normalize(self, x_i_graph, ii_graph, default_value=0.0):
        # 1. 确保 x_i_graph 边权为 1（如果原始边权不是 1，需要先二值化）
        x_i_binary = x_i_graph.copy()
        x_i_binary.data = np.ones_like(x_i_binary.data)  # 确保原始边权为 1
        
        # 2. 计算乘积矩阵（U×I）@（I×I）= U×I，非零值设为默认值
        multiplied_csr = x_i_graph @ ii_graph
        multiplied_csr.data = np.full_like(multiplied_csr.data, default_value)  # 关键：直接赋值data
        
        # 3. 矩阵相加（利用 CSR 加法的高效合并）
        combined_csr = multiplied_csr + x_i_binary  # 自动合并相同位置的元素
        
        # 4. Laplace 归一化（优化 CSR 路径）
        return laplace_transform(combined_csr)

    def get_propagation_graph_with_ii(self, bipartite_graph, ii_graph, modification_ratio=0):
        # 获取设备信息
        device = self.device
        num_part_1 = bipartite_graph.shape[0]
        num_items = ii_graph.shape[0]
        if ii_graph.shape[0] != bipartite_graph.shape[1] or ii_graph.shape[1] != num_items:
            raise ValueError(f"ii_graph 的大小 ({ii_graph.shape}) 与 bipartite_graph 中的物品数量不匹配，预期大小应为 ({bipartite_graph.shape[1]}, {bipartite_graph.shape[1]})。")

        # 构建传播图，同时填充 II 图
        upper_left = sp.csr_matrix((num_part_1, num_part_1))
        upper_right = bipartite_graph
        lower_left = bipartite_graph.T
        lower_right = ii_graph

        propagation_graph = sp.bmat([[upper_left, upper_right], [lower_left, lower_right]])

        # 如果修改比率不为 0
        if modification_ratio != 0:
            # 如果增强类型是 ED（边丢弃）
            if self.conf["aug_type"] == "ED":
                # 将传播图转换为 COO 格式
                graph = propagation_graph.tocoo()
                # 对边进行随机丢弃
                values = np_edge_dropout(graph.data, modification_ratio)
                # 重新构建传播图
                propagation_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        # 对传播图进行拉普拉斯变换并转换为张量，然后移动到指定设备
        return to_tensor(laplace_transform(propagation_graph)).to(device)


    # 获取传播图
    def get_propagation_graph(self, bipartite_graph, modification_ratio=0):
        # 获取设备信息
        device = self.device
        # 构建传播图，将二分图与其转置组合
        propagation_graph = sp.bmat([[sp.csr_matrix((bipartite_graph.shape[0], bipartite_graph.shape[0])), bipartite_graph], 
                                    [bipartite_graph.T, sp.csr_matrix((bipartite_graph.shape[1], bipartite_graph.shape[1]))]])

        # 如果修改比率不为 0
        if modification_ratio != 0:
            # 如果增强类型是 ED（边丢弃）
            if self.conf["aug_type"] == "ED":
                # 将传播图转换为 COO 格式
                graph = propagation_graph.tocoo()
                # 对边进行随机丢弃
                values = np_edge_dropout(graph.data, modification_ratio)
                # 重新构建传播图
                propagation_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        # 对传播图进行拉普拉斯变换并转换为张量，然后移动到指定设备
        return to_tensor(laplace_transform(propagation_graph)).to(device)

    # 获取聚合图
    def get_aggregation_graph(self, bipartite_graph, modification_ratio=0):
        # 获取设备信息
        device = self.device

        # 如果修改比率不为 0
        if modification_ratio != 0:
            # 如果增强类型是 ED（边丢弃）
            if self.conf["aug_type"] == "ED":
                # 将二分图转换为 COO 格式
                graph = bipartite_graph.tocoo()
                # 对边进行随机丢弃
                values = np_edge_dropout(graph.data, modification_ratio)
                # 重新构建二分图
                bipartite_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        # 计算每个节点的度，并加上一个小的常数防止除零错误
        bundle_size = bipartite_graph.sum(axis=1) + 1e-8
        # 对二分图进行归一化处理
        bipartite_graph = sp.diags(1/bundle_size.A.ravel()) @ bipartite_graph
        # 将处理后的二分图转换为张量并移动到指定设备
        return to_tensor(bipartite_graph).to(device)

    def propagate_ii(self, graph, item_feature, layer_coef, test):
        # 存储每一层的特征
        all_features = [item_feature]

        # 进行多层传播
        for i in range(self.num_layers):
            # 通过图卷积更新特征
            item_feature = torch.spmm(graph, item_feature)
            # 对特征进行 L2 归一化
            all_features.append(F.normalize(item_feature, p=2, dim=1))

        # 将每一层的特征堆叠起来，并乘以层融合权重
        all_features = torch.stack(all_features, 1) * layer_coef
        # 对每一层的特征进行求和
        all_features = torch.sum(all_features, dim=1)

        return all_features

    # 进行图的传播操作 ？二级嵌入
    def propagate(self, graph, A_feature, B_feature, graph_type, layer_coef, test):
        # 将 A 特征和 B 特征拼接在一起
        features = torch.cat((A_feature, B_feature), 0)
        # 存储每一层的特征
        all_features = [features]

        # 进行多层传播
        for i in range(self.num_layers):
            # 通过图卷积更新特征
            features = torch.spmm(graph, features) ## 稀疏矩阵和密集矩阵乘法 交互稀疏矩阵和第一级嵌入(torch) 后面的features是embedding矩阵而不是嵌入后的结果 前面的是EU,EB
            # 如果增强类型是 MD 且不在测试阶段
            if self.conf["aug_type"] == "MD" and not test:
                # 获取对应的丢弃层
                mess_dropout = self.mess_dropout_dict[graph_type]
                # 应用丢弃操作
                features = mess_dropout(features)
            # 如果增强类型是 Noise 且不在测试阶段
            elif self.conf["aug_type"] == "Noise" and not test:
                # 生成随机噪声
                random_noise = torch.rand_like(features).to(self.device)
                # 获取对应的噪声参数
                eps = self.eps_dict[graph_type]
                # 添加噪声
                features += torch.sign(features) * F.normalize(random_noise, dim=-1) * eps

            # 对特征进行 L2 归一化
            all_features.append(F.normalize(features, p=2, dim=1))

        # 将每一层的特征堆叠起来，并乘以层融合权重
        all_features = torch.stack(all_features, 1) * layer_coef
        # 对每一层的特征进行求和
        all_features = torch.sum(all_features, dim=1)
        # 分离 A 特征和 B 特征
        A_feature, B_feature = torch.split(all_features, (A_feature.shape[0], B_feature.shape[0]), 0)

        return A_feature, B_feature

    # 进行图的聚合操作 ## indirect
    def aggregate(self, agg_graph, node_feature, graph_type, test):
        # 通过矩阵乘法进行聚合操作
        aggregated_feature = torch.matmul(agg_graph, node_feature)

        # 如果增强类型是 MD 且不在测试阶段
        if self.conf["aug_type"] == "MD" and not test:
            # 获取对应的丢弃层
            mess_dropout = self.mess_dropout_dict[graph_type]
            # 应用丢弃操作
            aggregated_feature = mess_dropout(aggregated_feature)
        # 如果增强类型是 Noise 且不在测试阶段
        elif self.conf["aug_type"] == "Noise" and not test:
            # 生成随机噪声
            random_noise = torch.rand_like(aggregated_feature).to(self.device)
            # 获取对应的噪声参数
            eps = self.eps_dict[graph_type]
            # 添加噪声
            aggregated_feature += torch.sign(aggregated_feature) * F.normalize(random_noise, dim=-1) * eps

        return aggregated_feature

    # 融合用户和捆绑包的特征
    def fuse_users_bundles_feature(self, users_feature, bundles_feature):
        # 将用户特征堆叠起来
        users_feature = torch.stack(users_feature, dim=0)
        # 将捆绑包特征堆叠起来
        bundles_feature = torch.stack(bundles_feature, dim=0)

        # 模态聚合，根据模态融合权重对用户特征进行加权求和
        users_rep = torch.sum(users_feature * self.modal_coefs, dim=0)
        # 模态聚合，根据模态融合权重对捆绑包特征进行加权求和
        bundles_rep = torch.sum(bundles_feature * self.modal_coefs, dim=0)

        return users_rep, bundles_rep

    # 获取多模态表示
    def get_multi_modal_representations(self, test=False):
        #  =============================  UB graph propagation  =============================
        if test:
            # 在测试阶段，使用无丢弃的传播图进行传播
            UB_users_feature, UB_bundles_feature = self.propagate(self.UB_propagation_graph_ori, self.users_feature, self.bundles_feature, "UB", self.UB_layer_coefs, test)
        else:
            # 在训练阶段，使用带有丢弃的传播图进行传播
            UB_users_feature, UB_bundles_feature = self.propagate(self.UB_propagation_graph, self.users_feature, self.bundles_feature, "UB", self.UB_layer_coefs, test)

        #  =============================  UI graph propagation  =============================
        if test:
            # 在测试阶段，使用无丢弃的传播图进行传播
            UI_users_feature, UI_items_feature = self.propagate(self.UI_propagation_graph_ori, self.users_feature, self.items_feature, "UI", self.UI_layer_coefs, test)
            # 在测试阶段，使用无丢弃的聚合图进行聚合
            UI_bundles_feature = self.aggregate(self.BI_aggregation_graph_ori, UI_items_feature, "BI", test)
        else:
            # 在训练阶段，使用带有丢弃的传播图进行传播
            UI_users_feature, UI_items_feature = self.propagate(self.UI_propagation_graph, self.users_feature, self.items_feature, "UI", self.UI_layer_coefs, test)
            # 在训练阶段，使用带有丢弃的聚合图进行聚合
            UI_bundles_feature = self.aggregate(self.BI_aggregation_graph, UI_items_feature, "BI", test)
        #  =============================  BI graph propagation  =============================
        if test:
            # 测试阶段使用无丢弃的传播图进行捆绑包 - 物品图的传播
            BI_bundles_feature, BI_items_feature = self.propagate(self.BI_propagation_graph_ori, UI_bundles_feature, self.items_feature, "BI", self.BI_layer_coefs, test)
            # 测试阶段使用无丢弃的聚合图从物品特征聚合得到用户特征
            BI_users_feature = self.aggregate(self.UI_aggregation_graph_ori, BI_items_feature, "UI", test)
        else:
            # 训练阶段使用带丢弃的传播图进行捆绑包 - 物品图的传播
            BI_bundles_feature, BI_items_feature = self.propagate(self.BI_propagation_graph, UI_bundles_feature, self.items_feature, "BI", self.BI_layer_coefs, test)
            # 训练阶段使用带丢弃的聚合图从物品特征聚合得到用户特征
            BI_users_feature = self.aggregate(self.UI_aggregation_graph, BI_items_feature, "UI", test)
        
        # 收集三种图传播得到的用户特征
        users_feature = [UB_users_feature, UI_users_feature, BI_users_feature]
        # 收集三种图传播得到的捆绑包特征
        bundles_feature = [UB_bundles_feature, UI_bundles_feature, BI_bundles_feature]
        # 融合不同图得到的用户和捆绑包特征
        users_rep, bundles_rep = self.fuse_users_bundles_feature(users_feature, bundles_feature)

        return users_rep, bundles_rep

    # 计算对比损失
    def cal_c_loss(self, pos, aug):
        # pos: [batch_size, :, emb_size]，正样本特征
        # aug: [batch_size, :, emb_size]，增强样本特征
        # 提取正样本的第一个特征
        pos = pos[:, 0, :]
        # 提取增强样本的第一个特征
        aug = aug[:, 0, :]

        # 对正样本特征进行 L2 归一化
        pos = F.normalize(pos, p=2, dim=1)
        # 对增强样本特征进行 L2 归一化
        aug = F.normalize(aug, p=2, dim=1)
        # 计算正样本和增强样本的相似度得分
        pos_score = torch.sum(pos * aug, dim=1)  # [batch_size]
        # 计算正样本和所有增强样本之间的相似度得分矩阵
        ttl_score = torch.matmul(pos, aug.permute(1, 0))  # [batch_size, batch_size]

        # 对正样本相似度得分应用指数函数并除以温度参数
        pos_score = torch.exp(pos_score / self.c_temp)  # [batch_size]
        # 对所有相似度得分矩阵应用指数函数并按行求和
        ttl_score = torch.sum(torch.exp(ttl_score / self.c_temp), axis=1)  # [batch_size]

        # 计算对比损失
        c_loss = - torch.mean(torch.log(pos_score / ttl_score))

        return c_loss

    def load_item_relations(self):
        item_relations = {}
        with open('ii4.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line[:-1].split(' ')
                item1 = int(parts[0])
                item2 = int(parts[1])
                relation_type = float(parts[2])
                if relation_type == 3:## 已经尝试多种1 2配无效果 接下来尝试 1 和 3
                    if item1 not in item_relations:
                        item_relations[item1] = {'positive': [], 'negative': []}
                    item_relations[item1]['positive'].append(item2)
                elif relation_type == 4:
                    if item1 not in item_relations:
                        item_relations[item1] = {'positive': [], 'negative': []}
                    item_relations[item1]['negative'].append(item2)
        '''
        all_items = set(range(self.num_items))
        for item in item_relations:
            positive_items = set(item_relations[item]['positive'])
            negative_items = set(item_relations[item]['negative'])  
            # 如果需要补全负关系，可以使用以下代码
            # negative_items = negative_items.union(all_items - positive_items - {item})
            item_relations[item]['negative'] = list(negative_items)
        '''
        return item_relations

    def cal_ii_single_item_loss(self, k):
        # 从所有物品中随机抽取 k 个物品
        random_items = np.random.choice(self.num_items, k, replace=False)

        total_loss = 0
        valid_items_count = 0

        for item in random_items:
            if item in self.item_relations:
                # 获取当前物品的正样本
                positive_samples = self.item_relations[item]['positive']
                num_positive = len(positive_samples)

                if num_positive > 0:
                    # 获取当前物品的负样本
                    negative_candidates = self.item_relations[item]['negative']

                    if len(negative_candidates) >= num_positive:
                        # 从负样本候选集中随机选择与正样本数量相同的负样本
                        negative_samples = np.random.choice(negative_candidates, num_positive, replace=False)

                        positive_samples = torch.tensor(positive_samples, device=self.device)
                        negative_samples = torch.tensor(negative_samples, device=self.device)

                        item_feature = F.normalize(self.items_feature[item], p=2, dim=0)
                        positive_features = F.normalize(self.items_feature[positive_samples], p=2, dim=1)
                        negative_features = F.normalize(self.items_feature[negative_samples], p=2, dim=1)

                        positive_scores = torch.sum(item_feature * positive_features, dim=1)
                        negative_scores = torch.sum(item_feature * negative_features, dim=1)

                        positive_scores = torch.exp(positive_scores / self.c_temp)
                        negative_scores = torch.exp(negative_scores / self.c_temp)

                        total_scores = torch.cat([positive_scores, negative_scores])
                        ii_loss = - torch.mean(torch.log(positive_scores / total_scores.sum()))
                        total_loss += ii_loss
                        valid_items_count += 1

        if valid_items_count == 0:
            return torch.tensor(0.0, device=self.device)
        else:
            return total_loss / valid_items_count

    # 计算总损失
    def cal_loss(self, users_feature, bundles_feature):
        # users_feature / bundles_feature: [bs, 1+neg_num, emb_size]
        # 计算用户特征和捆绑包特征的内积作为预测得分
        pred = torch.sum(users_feature * bundles_feature, 2)
        # 计算 BPR 损失
        bpr_loss = cal_bpr_loss(pred)

        # 计算用户视角的对比损失
        u_view_cl = self.cal_c_loss(users_feature, users_feature)
        # 计算捆绑包视角的对比损失
        b_view_cl = self.cal_c_loss(bundles_feature, bundles_feature)
        # 计算IIgraph的对比损失
        k = 0
        # 可根据需要调整 k 的值
        ii_single_item_loss = self.cal_ii_single_item_loss(k)

        # 存储对比损失
        # c_losses = [u_view_cl, b_view_cl, 0.5*ii_single_item_loss]
        c_losses = [u_view_cl, b_view_cl]


        # 计算平均对比损失
        c_loss = sum(c_losses) / len(c_losses)

        return bpr_loss, c_loss, ii_single_item_loss

    # 前向传播函数
    def forward(self, batch, ED_drop=False):
        # 边缘丢弃可以按批次或按轮次进行，由训练循环控制
        if ED_drop:
            # 重新生成用户 - 捆绑包图的传播图
            self.UB_propagation_graph = self.get_propagation_graph(self.H_iub, self.conf["UB_ratio"])
            # 重新生成用户 - 物品图的传播图
            self.UI_propagation_graph = self.get_propagation_graph(self.ui_graph, self.conf["UI_ratio"])
            # 重新生成用户 - 物品图的聚合图
            self.UI_aggregation_graph = self.get_aggregation_graph(self.ui_graph, self.conf["UI_ratio"])
            # 重新生成捆绑包 - 物品图的传播图
            self.BI_propagation_graph = self.get_propagation_graph(self.bi_graph, self.conf["BI_ratio"])
            # 重新生成捆绑包 - 物品图的聚合图
            self.BI_aggregation_graph = self.get_aggregation_graph(self.bi_graph, self.conf["BI_ratio"])

        # users: [bs, 1]，批量中的用户索引
        # bundles: [bs, 1+neg_num]，批量中的捆绑包索引
        users, bundles = batch
        # 获取多模态的用户和捆绑包表示
        users_rep, bundles_rep = self.get_multi_modal_representations()

        # 根据用户索引获取用户表示，并扩展维度以匹配捆绑包数量
        users_embedding = users_rep[users].expand(-1, bundles.shape[1], -1)
        # 根据捆绑包索引获取捆绑包表示
        bundles_embedding = bundles_rep[bundles]

        # 计算 BPR 损失和对比损失
        bpr_loss, c_loss, ii_c_loss = self.cal_loss(users_embedding, bundles_embedding)

        return bpr_loss, c_loss, ii_c_loss

    # 评估函数
    def evaluate(self, propagate_result, users):
        # 从传播结果中分离用户特征和捆绑包特征
        users_feature, bundles_feature = propagate_result
        # 计算用户和捆绑包之间的得分矩阵
        scores = torch.mm(users_feature[users], bundles_feature.t())
        return scores
