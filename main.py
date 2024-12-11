# coding=utf-8
# Author: Jung
# Time: 2024/5/31 17:07


import math

import torch
from torch.autograd import Variable
from torch.optim import Adam
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
from CTDGDcenter import *
from evaluation import *
import torch.nn.functional as F
import torch.nn as nn
from sklearn.mixture import GaussianMixture
import argparse

FType = torch.FloatTensor
LType = torch.LongTensor

DID = 0
np.random.seed(826)
torch.manual_seed(826)
torch.cuda.manual_seed(826)
torch.cuda.manual_seed(826)
torch.cuda.manual_seed_all(826)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
class ClusterAssignment(nn.Module):
    def __init__(self, cluster_number, embedding_dimension, alpha, cluster_centers=None):
        super(ClusterAssignment, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(self.cluster_number, self.embedding_dimension, dtype=torch.float).to(torch.float32)
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers.to(torch.float32)
        self.cluster_centers = nn.Parameter(initial_cluster_centers)# .to(device)

    def forward(self, inputs):
        norm_squared = torch.sum((inputs.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)
def compute_homophily_ratio(graph, labels):
    """
    计算图的节点同质性比例

    :param graph: networkx图对象
    :param labels: 节点的标签字典 {node: label}
    :return: 图的同质性比例
    """
    homophily_sum = 0
    for node in graph.nodes:
        neighbors = list(graph.neighbors(node))
        if neighbors:
            same_label_count = sum(1 for neighbor in neighbors if labels[neighbor] == labels[node])
            homophily_sum += same_label_count / len(neighbors)
    return homophily_sum / graph.number_of_nodes()


def compute_edge_homophily_ratio(graph, labels):
    """
    计算图的边同质性比例

    :param graph: networkx图对象
    :param labels: 节点的标签字典 {node: label}
    :return: 边同质性比例
    """
    same_label_edges = 0
    total_edges = graph.number_of_edges()

    for u, v in graph.edges:
        if labels[u] == labels[v]:
            same_label_edges += 1

    return same_label_edges / total_edges
class CTVGAE:
    def __init__(self, args):
        self.args = args
        self.the_data = args.dataset
        self.file_path = 'data/%s/%s.txt' % (self.the_data, self.the_data)
        self.feature_path = 'pretrain/%s_feature.emb' % self.the_data
        self.label_path = 'data/%s/node2label.txt' % self.the_data
        self.labels = self.read_label()
        self.emb_size = args.emb_size
        self.neg_size = args.neg_size
        self.hist_len = args.hist_len
        self.batch = args.batch_size
        self.clusters = args.clusters
        self.epochs = args.epoch
        self.best_acc = 0
        self.best_nmi = 0
        self.best_ari = 0
        self.best_f1 = 0

        self.best_kmens_acc = 0
        self.best_kmens_nmi = 0
        self.best_kmens_ari = 0
        self.best_kmens_f1 = 0

        self.data = CTDGData(self.file_path, self.neg_size, self.hist_len, self.feature_path)


        # print("node homophily ratio = {}".format(compute_homophily_ratio(self.data.G, self.labels)))
        # print("edge homophily ratio = {}".format(compute_edge_homophily_ratio(self.data.G, self.labels)))

        self.node_dim = self.data.get_node_dim()
        self.edge_num = self.data.get_edge_num()
        self.feature = self.data.get_feature()

        self.node_emb = Variable(torch.from_numpy(self.feature).type(FType).to(device), requires_grad=True)
        self.pre_emb = Variable(torch.from_numpy(self.feature).type(FType).to(device), requires_grad=False)
        self.delta = Variable((torch.zeros(self.node_dim) + 1.).type(FType).to(device), requires_grad=True)

        self.cluster_layer = Variable((torch.zeros(self.clusters, self.emb_size) + 1.).type(FType).to(device), requires_grad=True)
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        kmeans = KMeans(n_clusters=self.clusters, n_init=20)
        _ = kmeans.fit_predict(self.feature)
        self.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)

        self.v = 1.0
        self.batch_weight = math.ceil(self.batch / self.edge_num)

        self.loss = torch.FloatTensor()

        self.pi = nn.Parameter(torch.ones(self.clusters) / self.clusters, requires_grad=True)#.to(device)
        self.mu_c = nn.Parameter(torch.randn(self.clusters, self.feature.shape[1]), requires_grad=True)#.to(device)
        self.log_sigma2_c = nn.Parameter(torch.randn(self.clusters, self.feature.shape[1]), requires_grad=True)#.to(device)
        gmm = GaussianMixture(n_components=self.clusters, covariance_type='diag', reg_covar=1e-3) # 1e-5:other datasets   1e-4:school
        print('gaussian mixture fit predtict')
        _ = gmm.fit_predict(self.node_emb.cpu().detach().numpy())
        self.pi.data = torch.FloatTensor(gmm.weights_).to(device)
        self.mu_c.data = torch.FloatTensor(gmm.means_).to(device)
        self.log_sigma2_c.data = torch.log(torch.FloatTensor(gmm.covariances_)).to(device)
        init_pred, _ = self.predict(self.node_emb)
        acc, nmi, ari, f1 = end2end_eva(self.labels, init_pred)
        print("init gmm > acc:{}, nmi:{}, ari:{}, f1:{}".format(acc, nmi, ari, f1))

        self.pseudo_lable = init_pred
        self.pseudo_lable = torch.from_numpy(init_pred)

        self.assignment = ClusterAssignment(self.clusters, self.feature.shape[1], 1, torch.from_numpy(kmeans.cluster_centers_)).to(device)
        self.loss_kl = nn.KLDivLoss(reduction='batchmean')

        self.opt = Adam(lr=args.learning_rate, params=[self.node_emb, self.delta, self.cluster_layer, self.pi, self.mu_c, self.log_sigma2_c,
                             self.assignment.cluster_centers])

    def predict(self, z):
        pi = self.pi
        log_sigma2_c = self.log_sigma2_c
        mu_c = self.mu_c
        det = 1e-2
        yita_c = torch.exp(torch.log(pi.unsqueeze(0)) + self._gaussian_pdfs_log(z, mu_c, log_sigma2_c)) + det
        yita = yita_c.cpu().detach().numpy()
        return np.argmax(yita, axis=1), yita_c.detach()
    def _gaussian_pdfs_log(self, x, mus, log_sigma2s):
        G = []
        for c in range(self.clusters):
            G.append(self._gaussian_pdf_log(x, mus[c:c + 1, :], log_sigma2s[c:c + 1, :]).view(-1, 1))
        return torch.cat(G, 1)

    def _gaussian_pdf_log(self, x, mu, log_sigma2):
        c = -0.5 * torch.sum(np.log(np.pi * 2) + log_sigma2 + (x - mu).pow(2) / torch.exp(log_sigma2), 1)
        return c

    def read_label(self):
        n2l = dict()
        labels = []
        with open(self.label_path, 'r') as reader:
            for line in reader:
                parts = line.strip().split()
                n_id, l_id = int(parts[0]), int(parts[1])
                n2l[n_id] = l_id
        reader.close()
        for i in range(len(n2l)):
            labels.append(int(n2l[i]))
        return labels

    def kl_loss(self, z, p):
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        the_kl_loss = F.kl_div((q.log()), p, reduction='batchmean')  # l_clu
        return the_kl_loss

    def target_dis(self, emb):
        q = 1.0 / (1.0 + torch.sum(torch.pow(emb.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return q

    def forward(self, s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask):
        batch = s_nodes.size()[0]
        s_node_emb = self.node_emb.index_select(0, Variable(s_nodes.view(-1)))# .view(batch, -1) # 源节点
        t_node_emb = self.node_emb.index_select(0, Variable(t_nodes.view(-1))).view(batch, -1) # 目标节点
        h_node_emb = self.node_emb.index_select(0, Variable(h_nodes.view(-1))).view(batch, self.hist_len, -1) # history_nodes
        n_node_emb = self.node_emb.index_select(0, Variable(n_nodes.view(-1))).view(batch, self.neg_size, -1) # neg_nodes
        s_pre_emb = self.pre_emb.index_select(0, Variable(s_nodes.view(-1))).view(batch, -1)

        # CT-CAVAE
        pi = self.pi
        mean_c = self.mu_c
        logstd_c = self.log_sigma2_c
        p_c_z = self.assignment(s_node_emb)
        q_c_xa = torch.exp(torch.log(pi.unsqueeze(0)) + self._gaussian_pdfs_log(s_node_emb, mean_c, logstd_c).to(torch.float64)) + 1e-2
        q_c_xa = q_c_xa / (q_c_xa.sum(1).view(-1, 1))
        q_c_xa = q_c_xa.to(torch.float32)
        loss_clus = self.loss_kl(p_c_z, q_c_xa)
        ####

        s_node_emb = s_node_emb.view(batch, -1)

        #
        q = self.target_dis(s_pre_emb)
        s_kl_loss = self.kl_loss(s_node_emb, q)
        l_node = s_kl_loss

        # pseudo label generate


        pseudo = torch.zeros(size=[batch, self.clusters]).to(torch.float32).to(device)
        pseudo[np.arange(batch), self.pseudo_lable[s_nodes]] = 1
        pseudo[np.arange(batch), self.pseudo_lable[t_nodes]] = 1
        pseudo_s_t = pseudo @ pseudo.T
        pseudo[:] = 0
        if self.hist_len > 1:
            pseudo_s_h = torch.zeros(size=[self.hist_len, batch, batch])
            for hist_idx in range(self.hist_len):
                pseudo[torch.arange(batch), self.pseudo_lable[s_nodes]] = 1
                pseudo[torch.arange(batch), self.pseudo_lable[h_nodes[:,hist_idx]]] = 1
                aux_pseudo = pseudo @ pseudo.T
                aux_pseudo = aux_pseudo * h_time_mask[:,0]
                pseudo_s_h[hist_idx, :, :] = aux_pseudo
            pseudo[:] = 0
        else:
            pseudo[torch.arange(batch), self.pseudo_lable[s_nodes]] = 1
            pseudo[torch.arange(batch), self.pseudo_lable[h_nodes]] = 1
            pseudo_s_h = pseudo @ pseudo.T
            pseudo_s_h = pseudo_s_h * h_time_mask
        ################################

        new_st_adj = torch.cosine_similarity(s_node_emb, t_node_emb)  # [b]
        res_st_loss = torch.norm(pseudo_s_t - new_st_adj, p=2)

        new_sh_adj = torch.cosine_similarity(s_node_emb.unsqueeze(1), h_node_emb, dim=2)  # [b,h]
        new_sh_adj = new_sh_adj * h_time_mask
        new_sn_adj = torch.cosine_similarity(s_node_emb.unsqueeze(1), n_node_emb, dim=2)  # [b,n]


        if self.hist_len > 1:
            res_sh_loss = 0
            for hist_idx in range(self.hist_len):
                res_sh_loss += torch.norm(pseudo_s_h[hist_idx] - new_sh_adj[:, hist_idx], p=2)
        else:
            res_sh_loss = torch.norm(pseudo_s_h - new_sh_adj, p=2)
        #####################################

        ##########################
        res_sn_loss = torch.norm(0 - new_sn_adj, p=2, dim=0).sum(dim=0, keepdims=False)
        l_batch = res_st_loss + res_sh_loss + res_sn_loss

        l_framework = l_node + l_batch

        att = softmax(((s_node_emb.unsqueeze(1) - h_node_emb) ** 2).sum(dim=2).neg(), dim=1) # w

        p_mu = ((s_node_emb - t_node_emb) ** 2).sum(dim=1).neg() # 按元素取负数 | 计算eq.1
        p_alpha = ((h_node_emb - t_node_emb.unsqueeze(1)) ** 2).sum(dim=2).neg() # 历史 - 目标节点 >>> mu(i, y, t)

        delta = self.delta.index_select(0, Variable(s_nodes.view(-1))).unsqueeze(1)
        d_time = torch.abs(t_times.unsqueeze(1) - h_times)

        p_lambda = p_mu + (att * p_alpha * torch.exp(delta * Variable(d_time)) * Variable(h_time_mask)).sum(
            dim=1)


        n_mu = ((s_node_emb.unsqueeze(1) - n_node_emb) ** 2).sum(dim=2).neg()
        n_alpha = ((h_node_emb.unsqueeze(2) - n_node_emb.unsqueeze(1)) ** 2).sum(dim=3).neg()

        n_lambda = n_mu + (att.unsqueeze(2) * n_alpha * (torch.exp(delta * Variable(d_time)).unsqueeze(2)) * (
            Variable(h_time_mask).unsqueeze(2))).sum(dim=1)

        loss = -torch.log(p_lambda.sigmoid() + 1e-6) - torch.log(n_lambda.neg().sigmoid() + 1e-6).sum(dim=1)

        total_loss = loss.sum() + l_framework - loss_clus

        return total_loss

    def update(self, s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask):
        if torch.cuda.is_available():
            with torch.cuda.device(DID):
                self.opt.zero_grad()
                loss = self.forward(s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask)
                self.loss += loss.data
                loss.backward()
                self.opt.step()
        else:
            self.opt.zero_grad()
            loss = self.forward(s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask)
            self.loss += loss.data
            loss.backward()
            self.opt.step()

    def train(self):
        for epoch in range(self.epochs):
            self.loss = 0.0
            loader = DataLoader(self.data, batch_size=self.batch, shuffle=True, num_workers=1)
            for i_batch, sample_batched in enumerate(loader):
                if i_batch != 0:
                    sys.stdout.write('\r' + str(i_batch * self.batch) + '\tloss: ' + str(
                        self.loss.cpu().numpy() / (self.batch * i_batch)))
                    sys.stdout.flush()

                self.update(sample_batched['source_node'].type(LType).to(device),
                            sample_batched['target_node'].type(LType).to(device),
                            sample_batched['target_time'].type(FType).to(device),
                            sample_batched['neg_nodes'].type(LType).to(device),
                            sample_batched['history_nodes'].type(LType).to(device),
                            sample_batched['history_times'].type(FType).to(device),
                            sample_batched['history_masks'].type(FType).to(device))

            if self.the_data == 'arxivLarge' or self.the_data == 'arxivPhy' or self.the_data == 'arxivMath':
                acc, nmi, ari, f1 = 0, 0, 0, 0
            else:
                pred, _ = self.predict(self.node_emb)
                self.pseudo_lable = torch.from_numpy(pred)
                acc, nmi, ari, f1 = end2end_eva(self.labels, pred)
                # acc, nmi, ari, f1 = 0, 0, 0, 0
                k_means_acc, k_means_nmi, k_means_ari, k_means_f1 = eva(self.clusters, self.labels, self.node_emb.cpu())

            if nmi > self.best_nmi and epoch > 10:
                self.best_acc = acc
                self.best_nmi = nmi
                self.best_ari = ari
                self.best_f1 = f1


            if k_means_nmi > self.best_kmens_nmi and epoch > 10:
                self.best_kmens_acc = k_means_acc
                self.best_kmens_nmi = k_means_nmi
                self.best_kmens_ari = k_means_ari
                self.best_kmens_f1 = k_means_f1

            sys.stdout.write('\repoch %d: loss=%.4f  ' % (epoch, (self.loss.cpu().numpy() / len(self.data))))
            sys.stdout.write('end-to-end | ACC(%.4f) NMI(%.4f) ARI(%.4f) F1(%.4f); k-menas | ACC(%.4f) NMI(%.4f) ARI(%.4f) F1(%.4f) \n' % (acc, nmi, ari, f1, k_means_acc, k_means_nmi, k_means_ari, k_means_f1))
            sys.stdout.flush()

        print('Best end-to-end performance: ACC(%.4f) NMI(%.4f) ARI(%.4f) F1(%.4f)' %
              (self.best_acc, self.best_nmi, self.best_ari, self.best_f1))
        print('Best k-mean performance: ACC(%.4f) NMI(%.4f) ARI(%.4f) F1(%.4f)' %
              (self.best_kmens_acc, self.best_kmens_nmi, self.best_kmens_ari, self.best_kmens_f1))
if __name__ == '__main__':
    data = 'arXivAI'
    print(data)
    k_dict = {'arXivAI': 5, 'arXivCS': 40, 'arxivPhy': 53, 'arxivMath': 31, 'arxivLarge': 172, 'school': 9, 'dblp': 10,
              'brain': 10, 'patent': 6}

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=data)
    parser.add_argument('--clusters', type=int, default=k_dict[data])
    # dblp/10, arxivAI/5
    parser.add_argument('--epoch', type=int, default=100)
    # dblp/50, arxivAI/200
    parser.add_argument('--neg_size', type=int, default=2)
    parser.add_argument('--hist_len', type=int, default=1)
    # dblp/5, arxivAI/1
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--learning_rate', type=float, default=0.0002) # patent: 0.01 brain: =0.0002 school: 0.0001 dblp 0.001 arxivAI 0.0002
    parser.add_argument('--emb_size', type=int, default=128)
    args = parser.parse_args()

    the_train = CTVGAE(args)
    the_train.train()
