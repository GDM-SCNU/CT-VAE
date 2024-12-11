# coding=utf-8
# Author: Jung
# Time: 2024/6/1 19:13

from torch.utils.data import Dataset
import numpy as np
import sys
import networkx as nx

class CTDGData(Dataset):
    def __init__(self, file_path, neg_size, hist_len, feature_path):
        self.neg_size = neg_size  # 负样本大小
        self.hist_len = hist_len  # 历史长度
        self.feature_path = feature_path  # 特征路径
        self.max_d_time = -sys.maxsize  # 最大的时间戳

        self.NEG_SAMPLING_POWER = 0.75
        self.neg_table_size = int(1e8)

        self.node2hist = dict()
        self.node_set = set()
        self.degrees = dict()
        self.edge_num = 0

        edge_list = list()
        with open(file_path, 'r') as infile:
            for line in infile:
                self.edge_num += 1
                parts = line.split()
                s_node = int(parts[0])  # 源节点
                t_node = int(parts[1])  # 目标节点
                d_time = float(parts[2])  # 时间戳

                self.node_set.update([s_node, t_node])  # [源节点，目标节点]
                edge_list.append([s_node, t_node])
                edge_list.append([t_node, s_node])
                # 记录历史节点 {源节点:[目标节点, 时间戳]...;}
                if s_node not in self.node2hist:
                    self.node2hist[s_node] = list()
                self.node2hist[s_node].append((t_node, d_time))
                if t_node not in self.node2hist:
                    self.node2hist[t_node] = list()
                self.node2hist[t_node].append((s_node, d_time))

                # 记录节点的度
                if s_node not in self.degrees:
                    self.degrees[s_node] = 0
                if t_node not in self.degrees:
                    self.degrees[t_node] = 0
                self.degrees[s_node] += 1
                self.degrees[t_node] += 1

                if d_time > self.max_d_time:  # 更新最大时间戳
                    self.max_d_time = d_time

        ### 计算数据集的同质性
        self.G = nx.from_edgelist(edge_list)

        # 对历史事件按照时间排序
        self.node_dim = len(self.node_set)
        self.data_size = 0  # number of event
        for s in self.node2hist:
            hist = self.node2hist[s]  # 根据source node find his-infomation
            hist = sorted(hist, key=lambda x: x[1])  # 根据时间戳进行排序（升）
            self.node2hist[s] = hist  # 重新赋值
            self.data_size += len(self.node2hist[s])

        # 创建事件索引
        self.idx2source_id = np.zeros((self.data_size,), dtype=np.int32)  # 源节点
        self.idx2target_id = np.zeros((self.data_size,), dtype=np.int32)  # 目标节点在历史中的索引
        idx = 0
        for s_node in self.node2hist:
            for t_idx in range(len(self.node2hist[s_node])):
                self.idx2source_id[idx] = s_node
                self.idx2target_id[idx] = t_idx
                idx += 1

        # 读取特征
        self.node_features = self.get_feature()
        # 生成负样本
        self.neg_table = np.zeros((self.neg_table_size,))
        self.init_neg_table()


    def __getitem__(self, idx):
        s_node = self.idx2source_id[idx]
        t_idx = self.idx2target_id[idx]
        t_node = self.node2hist[s_node][t_idx][0]
        t_time = self.node2hist[s_node][t_idx][1]

        if t_idx - self.hist_len < 0:
            hist = self.node2hist[s_node][0:t_idx]
        else:
            hist = self.node2hist[s_node][t_idx - self.hist_len:t_idx] # 按照事件时间长度选择时段出现的事件

        hist_nodes = [h[0] for h in hist]
        hist_times = [h[1] for h in hist]

        his_node = np.zeros((self.hist_len,))
        his_node[:len(hist_nodes)] = hist_nodes # 源节点的历史节点
        his_time = np.zeros((self.hist_len,))
        his_time[:len(hist_times)] = hist_times

        # 标记, 当源节点不存在历史事件时，为0
        his_masks = np.zeros((self.hist_len,))
        his_masks[:len(hist_nodes)] = 1.

        neg_nodes = self.negative_sampling()

        sample = {
            'source_node': s_node, # 源节点
            'target_node': t_node, # 目标节点
            'target_time': t_time, # 时间戳
            'history_nodes': his_node, # 源节点的历史节点
            'history_times': his_time, # 源节点的历史节点时间戳
            'history_masks': his_masks, # 标记是否存在历史节点
            'neg_nodes': neg_nodes, # 负样本
        }
        return sample

    def get_node_dim(self):
        return self.node_dim

    def get_edge_num(self):
        return self.edge_num

    def get_feature(self):
        node_emb = dict()
        with open(self.feature_path, 'r') as reader:
            reader.readline()
            for line in reader:
                embeds = np.fromstring(line.strip(), dtype=float, sep=' ')
                node_id = embeds[0]
                node_emb[node_id] = embeds[1:]
            reader.close()
        feature = []
        for i in range(len(node_emb)):
            feature.append(node_emb[i])

        return np.array(feature)

    def get_max_d_time(self):
        return self.max_d_time

    def init_neg_table(self):
        tot_sum, cur_sum, por = 0., 0., 0.

        n_id = 0
        for k in range(self.node_dim):
            tot_sum += np.power(self.degrees[k], self.NEG_SAMPLING_POWER)
        for k in range(self.neg_table_size):
            if (k + 1.) / self.neg_table_size > por:
                cur_sum += np.power(self.degrees[n_id], self.NEG_SAMPLING_POWER)
                por = cur_sum / tot_sum
                n_id += 1
            self.neg_table[k] = n_id - 1

    def __len__(self):  # 要求实现 返回数据集的样本数量
        return self.data_size

    def negative_sampling(self):
        rand_idx = np.random.randint(0, self.neg_table_size, (self.neg_size,))
        sampled_nodes = self.neg_table[rand_idx]
        return sampled_nodes

if __name__ == "__main__":
    the_data = 'dblp'
    file_path = 'data/%s/%s.txt' % (the_data, the_data)
    feature_path = 'pretrain/%s_feature.emb' % the_data
    label_path = 'data/%s/node2label.txt' % the_data
    _ = CTDGData(file_path, 0, 2, feature_path)