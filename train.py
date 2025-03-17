#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import yaml
import json
import argparse
from tqdm import tqdm
from itertools import product
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.optim as optim
from utility import Datasets
from models.MultiCBR import MultiCBR

# 定义命令行参数解析函数
def get_cmd():
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    # 实验设置相关参数
    # 指定使用的 GPU 编号
    parser.add_argument("-g", "--gpu", default="0", type=str, help="which gpu to use")
    # 指定使用的数据集，可选值为 NetEase, iFashion
    parser.add_argument("-d", "--dataset", default="NetEase", type=str, help="which dataset to use, options: NetEase, iFashion")
    # 指定使用的模型，当前仅支持 MultiCBR
    parser.add_argument("-m", "--model", default="MultiCBR", type=str, help="which model to use, options: MultiCBR")
    # 附加信息，将追加到日志文件名中
    parser.add_argument("-i", "--info", default="", type=str, help="any auxilary info that will be appended to the log file name")
    # 解析命令行参数
    args = parser.parse_args()

    return args

# 主函数，程序入口
def main():
    # 从配置文件中加载配置信息
    conf = yaml.safe_load(open("./config.yaml"))
    print("load config file done!")

    # 获取命令行参数并转换为字典形式
    paras = get_cmd().__dict__
    # 获取数据集名称
    dataset_name = paras["dataset"]

    # 确保选择的模型在支持的列表中
    assert paras["model"] in ["MultiCBR"], "Pls select models from: MultiCBR"

    # 根据数据集名称获取对应的配置信息
    if "_" in dataset_name:
        conf = conf[dataset_name.split("_")[0]]
    else:
        conf = conf[dataset_name]
    # 将数据集名称和模型名称添加到配置信息中
    conf["dataset"] = dataset_name
    conf["model"] = paras["model"]
    # 加载数据集
    dataset = Datasets(conf)

    # 将命令行中的 GPU 编号和附加信息添加到配置信息中
    conf["gpu"] = paras["gpu"]
    conf["info"] = paras["info"]

    # 获取数据集的用户数量、捆绑包数量和物品数量
    conf["num_users"] = dataset.num_users
    conf["num_bundles"] = dataset.num_bundles
    conf["num_items"] = dataset.num_items

    # 设置可见的 GPU 设备
    os.environ['CUDA_VISIBLE_DEVICES'] = conf["gpu"]
    # 选择使用的设备（GPU 或 CPU）
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device('cuda')
    conf["device"] = device
    print(conf)

    # 对不同的超参数组合进行遍历
    for lr, l2_reg, UB_ratio, UI_ratio, BI_ratio, embedding_size, num_layers, c_lambda, c_temp in \
            product(conf['lrs'], conf['l2_regs'], conf['UB_ratios'], conf['UI_ratios'], conf['BI_ratios'], conf["embedding_sizes"], conf["num_layerss"], conf["c_lambdas"], conf["c_temps"]):
        # 定义日志文件、TensorBoard 运行记录和模型检查点的路径
        log_path = "./log/%s/%s" % (conf["dataset"], conf["model"])
        run_path = "./runs/%s/%s" % (conf["dataset"], conf["model"])
        checkpoint_model_path = "./checkpoints/%s/%s/model" % (conf["dataset"], conf["model"])
        checkpoint_conf_path = "./checkpoints/%s/%s/conf" % (conf["dataset"], conf["model"])
        # 创建必要的目录
        if not os.path.isdir(run_path):
            os.makedirs(run_path)
        if not os.path.isdir(log_path):
            os.makedirs(log_path)
        if not os.path.isdir(checkpoint_model_path):
            os.makedirs(checkpoint_model_path)
        if not os.path.isdir(checkpoint_conf_path):
            os.makedirs(checkpoint_conf_path)

        # 更新配置信息中的 L2 正则化系数和嵌入维度
        conf["l2_reg"] = l2_reg
        conf["embedding_size"] = embedding_size

        # 定义实验设置的相关信息列表
        settings = []
        if conf["info"] != "":
            settings += [conf["info"]]

        settings += [conf["aug_type"]]
        if conf["aug_type"] == "ED":
            settings += [str(conf["ed_interval"])]
        if conf["aug_type"] == "OP":
            # 当增强类型为 OP 时，确保 UB_ratio、UI_ratio 和 BI_ratio 为 0
            assert UB_ratio == 0 and UI_ratio == 0 and BI_ratio == 0

        settings += ["Neg_%d" % (conf["neg_num"]), str(conf["batch_size_train"]), str(lr), str(l2_reg),
                     str(embedding_size)]

        # 更新配置信息中的丢弃率和层数
        conf["UB_ratio"] = UB_ratio
        conf["UI_ratio"] = UI_ratio
        conf["BI_ratio"] = BI_ratio
        conf["num_layers"] = num_layers
        settings += [str(UB_ratio), str(UI_ratio), str(BI_ratio), str(num_layers)]
        settings += ["_".join([str(conf['fusion_weights']["modal_weight"]), str(conf['fusion_weights']["UB_layer"]),
                               str(conf['fusion_weights']["UI_layer"]), str(conf['fusion_weights']["BI_layer"])])]

        # 更新配置信息中的对比损失系数和温度参数
        conf["c_lambda"] = c_lambda
        conf["c_temp"] = c_temp
        settings += [str(c_lambda), str(c_temp)]

        # 将设置信息拼接成字符串
        setting = "_".join(settings)
        log_path = log_path + "/" + setting
        run_path = run_path + "/" + setting
        checkpoint_model_path = checkpoint_model_path + "/" + setting
        checkpoint_conf_path = checkpoint_conf_path + "/" + setting

        # 创建 TensorBoard 记录器
        run = SummaryWriter(run_path)

        # 根据配置信息创建模型
        if conf['model'] == 'MultiCBR':
            model = MultiCBR(conf, dataset.graphs).to(device)
        else:
            raise ValueError("Unimplemented model %s" % (conf["model"]))

        # 创建优化器
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=conf["l2_reg"])

        # 计算每个 epoch 的批次数
        batch_cnt = len(dataset.train_loader)
        # 计算测试间隔和边丢弃间隔对应的批次数
        test_interval_bs = int(batch_cnt * conf["test_interval"])
        ed_interval_bs = int(batch_cnt * conf["ed_interval"])

        # 初始化最佳指标和最佳性能
        best_metrics, best_perform = init_best_metrics(conf)
        best_epoch = 0
        # 开始训练循环
        for epoch in range(conf['epochs']):
            # 计算当前 epoch 的起始批次索引
            epoch_anchor = epoch * batch_cnt
            # 将模型设置为训练模式
            model.train(True)
            # 创建进度条
            pbar = tqdm(enumerate(dataset.train_loader), total=len(dataset.train_loader))

            for batch_i, batch in pbar:
                # 将模型设置为训练模式
                model.train(True)
                # 清空优化器的梯度
                optimizer.zero_grad()
                # 将批次数据移动到指定设备
                batch = [x.to(device) for x in batch]
                # 计算当前批次的全局索引
                batch_anchor = epoch_anchor + batch_i

                # 判断是否进行边丢弃操作
                ED_drop = False
                if conf["aug_type"] == "ED" and (batch_anchor + 1) % ed_interval_bs == 0:
                    ED_drop = True
                # 前向传播计算损失
                bpr_loss, c_loss = model(batch, ED_drop=ED_drop)
                # 计算总损失
                loss = bpr_loss + conf["c_lambda"] * c_loss
                # 反向传播计算梯度
                loss.backward()
                # 更新模型参数
                optimizer.step()

                # 分离损失张量，避免梯度计算
                loss_scalar = loss.detach()
                bpr_loss_scalar = bpr_loss.detach()
                c_loss_scalar = c_loss.detach()
                # 将损失信息写入 TensorBoard
                run.add_scalar("loss_bpr", bpr_loss_scalar, batch_anchor)
                run.add_scalar("loss_c", c_loss_scalar, batch_anchor)
                run.add_scalar("loss", loss_scalar, batch_anchor)

                # 更新进度条的描述信息
                pbar.set_description("epoch: %d, loss: %.4f, bpr_loss: %.4f, c_loss: %.4f" %(epoch, loss_scalar, bpr_loss_scalar, c_loss_scalar))

                # 达到测试间隔时进行测试
                if (batch_anchor + 1) % test_interval_bs == 0:
                    metrics = {}
                    # 在验证集上进行测试
                    metrics["val"] = test(model, dataset.val_loader, conf)
                    # 在测试集上进行测试
                    metrics["test"] = test(model, dataset.test_loader, conf)
                    # 记录指标信息，更新最佳指标和最佳性能
                    best_metrics, best_perform, best_epoch = log_metrics(conf, model, metrics, run, log_path, checkpoint_model_path, checkpoint_conf_path, epoch, batch_anchor, best_metrics, best_perform, best_epoch)

# 初始化最佳指标
def init_best_metrics(conf):
    best_metrics = {}
    best_metrics["val"] = {}
    best_metrics["test"] = {}
    for key in best_metrics:
        best_metrics[key]["recall"] = {}
        best_metrics[key]["ndcg"] = {}
    # 为每个 topk 值初始化最佳指标为 0
    for topk in conf['topk']:
        for key, res in best_metrics.items():
            for metric in res:
                best_metrics[key][metric][topk] = 0
    best_perform = {}
    best_perform["val"] = {}
    best_perform["test"] = {}

    return best_metrics, best_perform

# 将指标信息写入日志文件和 TensorBoard
def write_log(run, log_path, topk, step, metrics):
    # 获取当前时间
    curr_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # 获取验证集和测试集的指标
    val_scores = metrics["val"]
    test_scores = metrics["test"]

    # 将指标信息写入 TensorBoard
    for m, val_score in val_scores.items():
        test_score = test_scores[m]
        run.add_scalar("%s_%d/Val" %(m, topk), val_score[topk], step)
        run.add_scalar("%s_%d/Test" %(m, topk), test_score[topk], step)

    # 构建验证集和测试集的日志信息
    val_str = "%s, Top_%d, Val:  recall: %f, ndcg: %f" %(curr_time, topk, val_scores["recall"][topk], val_scores["ndcg"][topk])
    test_str = "%s, Top_%d, Test: recall: %f, ndcg: %f" %(curr_time, topk, test_scores["recall"][topk], test_scores["ndcg"][topk])

    # 打开日志文件并写入信息
    log = open(log_path, "a")
    log.write("%s\n" %(val_str))
    log.write("%s\n" %(test_str))
    log.close()

    # 打印日志信息
    print(val_str)
    print(test_str)

# 记录指标信息，更新最佳指标和最佳性能
def log_metrics(conf, model, metrics, run, log_path, checkpoint_model_path, checkpoint_conf_path, epoch, batch_anchor, best_metrics, best_perform, best_epoch):
    # 为每个 topk 值写入日志信息
    for topk in conf["topk"]:
        write_log(run, log_path, topk, batch_anchor, metrics)

    # 打开日志文件
    log = open(log_path, "a")

    topk_ = 20
    print("top%d as the final evaluation standard" %(topk_))
    # 判断当前验证集指标是否优于最佳指标
    if metrics["val"]["recall"][topk_] > best_metrics["val"]["recall"][topk_] and metrics["val"]["ndcg"][topk_] > best_metrics["val"]["ndcg"][topk_]:
        # 保存模型参数
        torch.save(model.state_dict(), checkpoint_model_path)
        # 保存配置信息
        dump_conf = dict(conf)
        del dump_conf["device"]
        json.dump(dump_conf, open(checkpoint_conf_path, "w"))
        # 更新最佳 epoch
        best_epoch = epoch
        # 获取当前时间
        curr_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # 更新最佳指标
        for topk in conf['topk']:
            for key, res in best_metrics.items():
                for metric in res:
                    best_metrics[key][metric][topk] = metrics[key][metric][topk]

            # 更新最佳性能信息
            best_perform["test"][topk] = "%s, Best in epoch %d, TOP %d: REC_T=%.5f, NDCG_T=%.5f" %(curr_time, best_epoch, topk, best_metrics["test"]["recall"][topk], best_metrics["test"]["ndcg"][topk])
            best_perform["val"][topk] = "%s, Best in epoch %d, TOP %d: REC_V=%.5f, NDCG_V=%.5f" %(curr_time, best_epoch, topk, best_metrics["val"]["recall"][topk], best_metrics["val"]["ndcg"][topk])
            # 打印最佳性能信息
            print(best_perform["val"][topk])
            print(best_perform["test"][topk])
            # 将最佳性能信息写入日志文件
            log.write(best_perform["val"][topk] + "\n")
            log.write(best_perform["test"][topk] + "\n")

    # 关闭日志文件
    log.close()

    return best_metrics, best_perform, best_epoch

# 在测试集或验证集上进行测试
def test(model, dataloader, conf):
    tmp_metrics = {}
    # 初始化临时指标字典
    for m in ["recall", "ndcg"]:
        tmp_metrics[m] = {}
        for topk in conf["topk"]:
            tmp_metrics[m][topk] = [0, 0]

    # 获取设备信息
    device = conf["device"]
    # 将模型设置为评估模式
    model.eval()
    # 获取多模态表示
    rs = model.get_multi_modal_representations(test=True) ##user的feature组和bundle的feature组
    # 遍历数据加载器
    for users, ground_truth_u_b, train_mask_u_b in dataloader:
        # 进行评估，得到预测结果
        pred_b = model.evaluate(rs, users.to(device))
        # 排除训练集中已有的交互
        pred_b -= 1e8 * train_mask_u_b.to(device)
        # 计算指标
        tmp_metrics = get_metrics(tmp_metrics, ground_truth_u_b.to(device), pred_b, conf["topk"])

    metrics = {}
    # 计算最终指标
    for m, topk_res in tmp_metrics.items():
        metrics[m] = {}
        for topk, res in topk_res.items():
            metrics[m][topk] = res[0] / res[1]

    return metrics


# 计算指标
def get_metrics(metrics, grd, pred, topks):
    tmp = {"recall": {}, "ndcg": {}}
    # 为每个 topk 值计算召回率和 NDCG
    for topk in topks:## topk列表的个数
        # 获取每个用户的前 topk 个预测结果的索引
        _, col_indice = torch.topk(pred, topk)
        # 生成对应的行索引
        row_indice = torch.zeros_like(col_indice) + torch.arange(pred.shape[0], device=pred.device, dtype=torch.long).view(-1, 1)##？没搞懂，似乎是返回用户对应行号
        
        # 判断预测结果是否命中真实结果
        is_hit = grd[row_indice.view(-1), col_indice.view(-1)].view(-1, topk)
        # 计算召回率
        tmp["recall"][topk] = get_recall(pred, grd, is_hit, topk)
        # 计算 NDCG
        tmp["ndcg"][topk] = get_ndcg(pred, grd, is_hit, topk)

    for m, topk_res in tmp.items():
        for topk, res in topk_res.items():
            for i, x in enumerate(res):
                metrics[m][topk][i] += x

    return metrics


def get_recall(pred, grd, is_hit, topk):
    epsilon = 1e-8
    # 计算每个样本在 topk 预测中命中真实正样本的数量
    hit_cnt = is_hit.sum(dim=1)
    # 计算每个样本的真实正样本数量
    num_pos = grd.sum(dim=1)
    num_pos = torch.tensor(num_pos, device=torch.device('cuda'))
    # 移除那些没有真实正样本的测试样本（避免除零错误）
    # 分母为总样本数减去没有正样本的样本数
    denorm = pred.shape[0] - (num_pos == 0).sum().item()
    # 分子为每个样本的召回率之和，召回率为命中数除以正样本数（加上一个小量 epsilon 避免除零）
    nomina = (hit_cnt / (num_pos + epsilon)).sum().item()

    return [nomina, denorm]


def get_ndcg(pred, grd, is_hit, topk):
    def DCG(hit, topk, device):
        # 计算折损累计增益（DCG）
        # 将命中情况（0 或 1）除以对数（以 2 为底）的位置（从 2 开始，因为第一个位置的对数为 0）
        hit = hit / torch.log2(torch.arange(2, topk + 2, device=device, dtype=torch.float))
        # 对每个样本的 DCG 求和
        return hit.sum(-1)

    def IDCG(num_pos, topk, device):
        # 计算理想折损累计增益（IDCG）
        hit = torch.zeros(topk, dtype=torch.float,device=device)## tokp大小的行向量
        # 将真实正样本数量范围内的位置设为 1
        hit[:num_pos] = 1##有多少个行向量就设置为多少，越多IDCG越高
        # 计算 IDCG
        return DCG(hit, topk, device)

    device = grd.device
    IDCGs = torch.empty(1 + topk, dtype=torch.float, device=device)
    IDCGs[0] = 1  # 避免 0/0 的情况
    for i in range(1, topk + 1):
        # 预先计算不同正样本数量下的 IDCG
        IDCGs[i] = IDCG(i, topk, device)

    # 限制真实正样本数量在 0 到 topk 之间，并转换为长整型
    num_pos = grd.sum(dim=1).clamp(0, topk).to(torch.long)
    # 计算当前预测的 DCG
    dcg = DCG(is_hit, topk, device)

    # 根据真实正样本数量获取对应的 IDCG
    idcg = IDCGs[num_pos]
    # 计算归一化折损累计增益（NDCG）
    ndcg = dcg / idcg.to(device)##is_hit()/num_pos(总共实际)
    # 分母为总样本数减去没有正样本的样本数
    denorm = pred.shape[0] - (num_pos == 0).sum().item()
    # 分子为每个样本的 NDCG 之和
    nomina = ndcg.sum().item()

    return [nomina, denorm]


if __name__ == "__main__":
    main()
