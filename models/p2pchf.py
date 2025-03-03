import math
import scipy.io
import torch.optim as optim
import torch.nn as nn
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
from torch import cosine_similarity
from tqdm import tqdm
import copy
from utils.args import *
from models.utils.federated_model import FederatedModel
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd
import torch
from utils.util import create_if_not_exists
from torch.nn.functional import normalize


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Federated learning via FCCLPlus.')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class P2PCHF(FederatedModel):
    NAME = 'p2pchf'
    COMPATIBILITY = ['homogeneity', 'heterogeneity']

    def __init__(self, nets_list, args, ):
        super(P2PCHF, self).__init__(nets_list, args, )
        self.T=1
        self.l2norm = Normalize(2)
        self.KLD_criterion = KLD().cuda()
        self.public_lr = args.public_lr
        self.public_epoch = args.public_epoch
        self.prev_nets_list = []
        self.criterionCE = nn.CrossEntropyLoss().to(self.device)
        self.InstanceLoss = InstanceLoss(batch_size=64, temperature=1, device=self.device)
        self.ClusterLoss=ClusterLoss(class_num=5,temperature=1, device=self.device)
        self.DistillKL_logit_stand=DistillKL_logit_stand().to(self.device)
        self.mmd = nn.L1Loss(reduction='mean')


    def ini(self):

        for j in range(self.args.parti_num):
            self.prev_nets_list.append(copy.deepcopy(self.nets_list[j]))
        if self.args.structure == 'homogeneity':  # 同构
            self.global_net = copy.deepcopy(self.nets_list[0])
            global_w = self.nets_list[0].state_dict()
            for _, net in enumerate(self.nets_list):
                net.load_state_dict(global_w)
        else:
            pass

    # 协作更新
    def col_update(self, communication_idx, publoader):
        epoch_loss_dict = {}
        tsne_linear_output_list_0,tsne_linear_output_list_1,tsne_linear_output_list_2,tsne_linear_output_list_3=[],[],[],[]
        for pub_epoch_idx in range(self.public_epoch):
            for batch_idx, (public_data, y,train_domain_labels) in enumerate(publoader):
                batch_loss_dict = {}  # 每一个batch的损失字典
                linear_output_list = []  # logist输出
                linear_output_target_list = []  # logist输出(克隆)

                logits_attention_target_list = []
                logitis_attentin_list = []
                public_data = public_data.to(self.device)
                # 以此取出模型，输入公共数据
                for net_id, net in enumerate(self.nets_list):
                    net = net.to(self.device)
                    net.train()
                    '''
                    FISL Loss for overall Network
                    '''
                    linear_output=net(public_data)
                    linear_output=F.softmax(linear_output,dim=1)
                    linear_output_target_list.append(linear_output.clone().detach())  # 克隆数据，并且不会回传梯度(detach)
                    linear_output_list.append(linear_output)
                    '''
                    logits_alignment类对齐损失
                    '''
                    logits_attention= net.features_attention(public_data)
                    logits_attention = normalize(logits_attention, dim=1)
                    logits_attention_target_list.append(logits_attention.clone().detach())
                    logitis_attentin_list.append(logits_attention)

                    # pre_train_feature[net_id]=tsne_linear_output
                for net_idx, net in enumerate(self.nets_list):
                    optimizer = optim.Adam(net.parameters(), lr=self.public_lr)
                    '''
                   logits_alignment类对齐损失
                    '''
                    linear_output = linear_output_list[net_idx]  # 实际每个客户端输出的logist，带有梯度回传的
                    linear_output_target_avg_list = []
                    for k in range(self.args.parti_num):
                         if  net_idx!=k:
                            linear_output_target_avg_list.append(linear_output_target_list[k])

                    linear_output_target_avg = torch.mean(torch.stack(linear_output_target_avg_list), 0)




                    # 每个模型和平均值的相似度
                    z_1_bn = (linear_output - linear_output.mean(0)) / linear_output.std(0)  ##标准化
                    z_2_bn = (linear_output_target_avg - linear_output_target_avg.mean(0)) / linear_output_target_avg.std(0)  # 标准化
                    c = z_1_bn.T @ z_2_bn
                    c.div_(len(public_data))
                    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()  # 同纬度相关
                    fccl_loss =on_diag+self.ClusterLoss(linear_output)
                    '''
                    FISL实例对齐损失
                    '''
                    logits_attention = logitis_attentin_list[net_idx]  # 客户端的
                    linear_attention_target_avg_list = []
                    for k in range(self.args.parti_num):
                        if net_idx != k:
                            linear_attention_target_avg_list.append(logits_attention_target_list[k])
                    linear_attention_target_avg = torch.mean(torch.stack(linear_attention_target_avg_list), 0)
                    loss_attention,c_array= self.InstanceLoss(logits_attention, linear_attention_target_avg,self.args.fccm)
                    '''
                    #反向传播
                    '''
                    optimizer.zero_grad()
                    col_loss=0
                    if self.args.model_setting==2:
                        col_loss = loss_attention * self.args.fisl
                    elif self.args.model_setting==3:
                        col_loss= fccl_loss + loss_attention * self.args.fisl
                    elif self.args.model_setting==4:
                        col_loss = fccl_loss + loss_attention * self.args.fisl
                    elif self.args.model_setting == 5:#RHFL
                        for tec in linear_output_target_avg_list:
                            col_loss+=self.KLD_criterion(tec,linear_output)
                    elif self.args.model_setting == 6:#FedDF
                        linear_output_target_avg = torch.mean(torch.stack(linear_output_target_list), 0)
                        col_loss=self.KLD_criterion(linear_output_target_avg,linear_output )
                    elif self.args.model_setting == 7:#FedMD
                        linear_output_target_avg = torch.mean(torch.stack(linear_output_target_list), 0)
                        col_loss= self.mmd(linear_output,linear_output_target_avg )
                    elif self.args.model_setting == 8:#FML
                        col_loss = fccl_loss + loss_attention * self.args.fisl
                    elif self.args.model_setting == 9:#FedMatch
                        for tec in linear_output_target_avg_list:
                            col_loss += self.KLD_criterion(tec, linear_output)
                        teu_sum=torch.sum(torch.stack(linear_output_target_avg_list), dim=0)
                        result = torch.zeros_like(teu_sum)
                        max_indices = torch.argmax(teu_sum, dim=1)
                        result[torch.arange(teu_sum.size(0)), max_indices] = 1
                        # 其进行四舍五入并保留三位小数
                    batch_loss_dict[net_idx] = {'fccl': round(fccl_loss.item(), 3),
                                                'logits_attention': round(loss_attention.item(), 3)}
                    if batch_idx == len(publoader) - 2:
                        print('Communcation' + str(communication_idx) + ': Net' + str(net_idx) +
                              ': Instance_alignment: ' + str(round(loss_attention.item(), 3)) + ' logits_alignment: ' + str(
                            round(fccl_loss.item(), 3)))
                    col_loss.backward()
                    optimizer.step()
                epoch_loss_dict[batch_idx] = batch_loss_dict  # 这一轮协同更新第几个batch对应的每个客户端的损失
        if tsne_linear_output_list_0 and tsne_linear_output_list_1 and tsne_linear_output_list_2 and tsne_linear_output_list_3:
            self._tsne_plot(tsne_linear_output_list_0,tsne_linear_output_list_1,tsne_linear_output_list_2,tsne_linear_output_list_3,communication_idx)
        return None


    # 本地更新：蒸馏
    def loc_update(self,epoch_index, priloader_list):
        global target, source, sourc_id, target_id
        if self.args.structure == 'homogeneity':
            self.aggregate_nets(None)
        linear_output_target_list=[]
        KL_list=[]
        for i in range(self.args.parti_num):
            # 学生模型,教师模型，私有数据训练集
            linear_out,average_KL_loss=self._train_net(i, self.nets_list[i], self.prev_nets_list[i], priloader_list[i],epoch_index)
            linear_output_target_list.append(linear_out)
            KL_list.append(average_KL_loss)
        self.copy_nets2_prevnets()
        return KL_list

    def _train_net(self, index, net,inter_net, train_loader,epoch_index):

        global linear_out
        a=self.args.temp
        net = net.to(self.device)  # 学生模型
        inter_net = inter_net.to(self.device)  # 教师模型
        optimizer = optim.Adam(net.parameters(), lr=self.local_lr)
        # 分类交叉熵
        criterionCE = nn.CrossEntropyLoss().to(self.device)
        criterionKL = nn.KLDivLoss(reduction='batchmean')
        criterionKL.to(self.device)
        # 学生模型与教师模型之间KL散度
        iterator = tqdm(range(self.local_epoch))
        KL_loss_list_epoch = []
        for _ in iterator:
            KL_loss_list = []
            for batch_idx, (private_data, labels, domian_labels) in enumerate(train_loader):
                private_data = private_data.to(self.device)
                labels = labels.to(self.device)
                outputs = net(private_data)  # (128,10)
                linear_out=outputs.detach().cpu()
                s_anchors=self.compute_class_centroids(self.l2norm(linear_out),labels.clone().detach().cpu())
                s_anchors=self.l2norm(s_anchors).cuda()
                stu_contrastive = torch.div(torch.mm(outputs, s_anchors.T),self.T)

                # 冻结，梯度消失
                with torch.no_grad():
                    inter_outputs = inter_net(private_data)
                    labels_t=torch.argmax(inter_outputs, dim=1)
                    linear_out = inter_outputs.detach().cpu()
                    t_anchors = self.compute_class_centroids(self.l2norm(linear_out), labels.clone().detach().cpu())
                    t_anchors = self.l2norm( t_anchors).cuda()
                    tea_contrastive = torch.div(torch.mm(inter_outputs,  t_anchors.T), self.T)



                loss_hard = criterionCE(outputs, labels)
                inter_loss = self.DistillKL_logit_stand(outputs, inter_outputs, 1)
                crp_loss=self.KLD_criterion(tea_contrastive, stu_contrastive)
                loss_hard_t = criterionCE(outputs, labels_t)
                loss_rem=a*inter_loss+(1-a)*loss_hard_t+crp_loss*0.5
                loss = loss_hard+loss_rem
                optimizer.zero_grad()
                KL_loss_list.append(inter_loss.item())
                loss.backward()
                #更新迭代器的描述信息，显示当前本地参与者的索引号和损失值
                if self.args.model_setting ==4 :
                    iterator.desc = "Local Pariticipant %d losskd = %0.3f KL = %0.6f" % (index, loss_hard.item(),inter_loss.item())

                else:
                    iterator.desc = "Local Pariticipant %d lossCE = %0.3f KL = %0.3f  " % (index, loss_hard.item(),inter_loss.item())

                optimizer.step()

        average_KL_loss=np.mean(KL_loss_list)
        return linear_out,average_KL_loss


    def compute_class_centroids(self,features, labels):
        """
        计算每个类别的质心。
        :param features: (batch_size, num_features) 的张量，包含特征向量
        :param labels: (batch_size,) 的张量，包含对应的类别标签
        :return: 一个字典，键是类别标签，值是对应的质心
        """
        unique_labels = torch.unique(labels)  # 获取唯一的类别标签
        anchors = {}
        for label in unique_labels:
            # 获取属于当前类别的样本的特征向量
            class_features = features[labels == label]
            # 计算类别中心（均值）作为锚点
            class_anchor = torch.mean(class_features, dim=0)
            anchors[label.item()] = class_anchor
        # 将 labels 转换为列表以便于索引
        labels_list = labels.tolist()
        # 将每个样本对应的质心按照其类别标签进行索引
        selected_anchors = torch.stack([anchors[label] for label in labels_list])
        return selected_anchors
        #计算输入样本的类质心

    def contrastive_loss(self,features, labels, centroids, margin=0.1):
        """
        使用类质心作为锚点的对比损失函数。
        :param features: (batch_size, num_features) 的张量，包含特征向量
        :param labels: (batch_size,) 的张量，包含对应的类别标签
        :param centroids: 类别质心的字典
        :param margin: 对比损失中的间隔
        :return: 对比损失值
        """
        loss = 0.0
        for i, feature in enumerate(features):
            # 获取当前样本的类别标签
            label = labels[i].item()
            # 获取当前样本的质心（锚点）
            positive_centroid = centroids[label]

            # 计算与正锚点的相似度（例如，余弦相似度）
            # positive_similarity = F.cosine_similarity(feature.unsqueeze(0), positive_centroid.unsqueeze(0), dim=1)
            dot_product = torch.dot(feature, torch.tensor(positive_centroid).cuda())
            # 计算范数（模）
            norm_sample = torch.norm(feature)
            norm_anchor = torch.norm(torch.tensor(positive_centroid))
            # 计算余弦相似度
            positive_similarity = dot_product / (norm_sample * norm_anchor)

            # 计算与其他所有锚点的相似度
            negative_similarities = []
            for other_label, other_centroid in centroids.items():
                if other_label != label:
                    dot_product = torch.dot(feature, torch.tensor(other_centroid).cuda())
                    # 计算范数（模）
                    norm_sample = torch.norm(feature)
                    norm_anchor = torch.norm(torch.tensor(other_centroid))
                    # 计算余弦相似度
                    negative_similarity= dot_product / (norm_sample * norm_anchor)
                    negative_similarities.append(negative_similarity)
                    # 将列表转换为张量
            negative_similarities = torch.stack(negative_similarities)

            # 计算损失（例如，triplet loss变种）
            # 确保正锚点的相似度大于所有负锚点的相似度，至少大于margin
            loss_i = F.relu(1 - positive_similarity + negative_similarities.max() + margin).mean()
            loss += loss_i

            # 平均损失
        loss /= features.size(0)
        return loss

        # 示例：



class InstanceLoss(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super(InstanceLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)

        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j,fccm):
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)  # （128,64）

        sim = torch.matmul(z, z.T) / self.temperature
        a=sim.cpu().detach().numpy()
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)  # (batchsize,1)
        negative_samples = sim[self.mask].reshape(N, -1)

        logits = torch.cat((positive_samples, negative_samples), dim=1)#(128,127)
        on_diag = logits[:,0].add_(-1).pow_(2).sum()  # 同纬度相关
        off_diag = logits[:,1:].add_(1).pow_(2).sum()  # 不同纬度去相关
        # FCCM总损失
        loss = on_diag + fccm * off_diag
        loss /= N
        c_array=sim.detach().cpu().numpy()

        return loss,c_array

class ClusterLoss(nn.Module):
    def __init__(self, class_num, temperature, device):
        super(ClusterLoss, self).__init__()
        self.class_num = class_num
        self.temperature = temperature
        self.device = device
        self.mask = self.mask_correlated_clusters(class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, c_i):
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        return ne_i




class DistillKL_logit_stand(nn.Module):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self):
        super(DistillKL_logit_stand, self).__init__()

    def normalize(self,logit):
        mean = logit.mean(dim=-1, keepdims=True)
        stdv = logit.std(dim=-1, keepdims=True)
        return (logit - mean) / (1e-7 + stdv)

    def forward(self, y_s, y_t, temp):
        T = temp

        KD_loss = 0
        KD_loss += nn.KLDivLoss(reduction='batchmean')(F.log_softmax(y_s / T, dim=1),
                                                       F.softmax(y_t / T, dim=1))*T*T


        return KD_loss

class KLD(nn.Module):

    def forward(self, targets, inputs):
        targets = F.softmax(targets, dim=1)
        inputs = F.log_softmax(inputs, dim=1)
        # print(targets, F.softmax(inputs, dim=1))

        return F.kl_div(inputs, targets, reduction='batchmean')
class Normalize(nn.Module):
    """normalization layer"""
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out