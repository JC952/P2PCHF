import torch
from argparse import Namespace
from datasets.FedLeBearing import FedLeaBearing_priv, FedLeaBearing_publ
from models.utils.federated_model import FederatedModel
from typing import Tuple
from utils.logger import CsvWriter
import numpy as np
def evaluate(model: FederatedModel, test_dl: FedLeaBearing_priv, setting: str) -> Tuple[list, list,dict]:
    intra_accs = []
    inter_accs = []
    test_len = len(test_dl)

    #领域内评估
    for i in range(model.args.parti_num):
        if setting == 'domain_skew':
            dl = test_dl[i % test_len]
        else:
            dl = test_dl
        net = model.nets_list[i]
        net = net.to(model.device)
        status = net.training
        net.eval()
        correct, total, top1 = 0.0, 0.0, 0.0
        #领域内评估
        for batch_idx, (private_test, labels,domian_labels) in enumerate(dl):
            with torch.no_grad():
                private_test, labels = private_test.to(model.device), labels.to(model.device)
                outputs = net(private_test)
                _, max_index = torch.topk(outputs, 1, dim=-1)
                labels = labels.view(-1, 1)
                top1 += (labels ==max_index[:, :]).sum().item()
                total += labels.size(0)

        top1acc = round(100 * top1 / total, 2)
        intra_accs.append(top1acc)
        #使模型回到训练模式
        net.train(status)
    # 领域间评估
    if setting == 'domain_skew':
        for i in range(model.args.parti_num):
            inter_net_accs = []
            net = model.nets_list[i]
            status = net.training
            net.eval()
            for j, dl in enumerate(test_dl):
                if i % test_len != j :
                    correct, total, class_right = 0.0, 0.0, 0.0,
                    for batch_idx, (private_test, labels,domian_labels) in enumerate(dl):
                        with torch.no_grad():
                            private_test, labels = private_test.to(model.device), labels.to(model.device)
                            outputs = net(private_test)
                            _, cls_pre = outputs.max(dim=1)
                            class_right += torch.sum(cls_pre == labels.data)
                            total += labels.size(0)
                    top1acc = round(100 * class_right.item() /len(dl.dataset), 2)
                    inter_net_accs.append(top1acc)
            inter_accs.append(np.mean(inter_net_accs))

            net.train(status)


    return intra_accs, inter_accs

def train(model: FederatedModel, public_dataset:  FedLeaBearing_publ, private_dataset:  FedLeaBearing_priv,
          args: Namespace,loop=0,method=0,data_name=0) -> None:
    global domains_list, best_epoch, model_dict_all
    if args.csv_log:
        csv_writer = CsvWriter(args, private_dataset)
    if hasattr(args, 'public_batch_size'):#检查是否有该属性
        pub_loader = public_dataset.get_data_loaders()

    if args.structure == 'homogeneity':
        pri_train_loaders, test_loaders = private_dataset.get_data_loaders()
    elif args.structure == 'heterogeneity':
        selected_domain_list = []
        domains_list = private_dataset.DOMAINS_LIST
        domains_len = len(domains_list)
        for i in range(len(domains_list)):
            index = i % domains_len
            selected_domain_list.append(domains_list[index])
            #私有模型的训练和测试数据集
        pri_train_loaders, test_loaders = private_dataset.get_data_loaders(selected_domain_list)
    if hasattr(model, 'ini'):
        model.ini()

    #领域内准确率和领域外准确率
    intra_accs_dict = {}
    inter_accs_dict = {}
    KL_loss_dict = {}
    mean_intra_acc_list = []
    mean_inter_acc_list = []
    Epoch = args.communication_epoch
    for epoch_index in range(Epoch):

        model.epoch_index = epoch_index

        if hasattr(args, 'public_batch_size') and args.model_setting!=1:
            model.col_update(epoch_index, pub_loader)
            model.public_lr = args.public_lr
        intra_accs, inter_accs = evaluate(model, test_loaders, private_dataset.setting)
        if hasattr(model, 'loc_update'):
            #私有数据训练集
            KL_list=model.loc_update(epoch_index,pri_train_loaders)
            KL_loss_dict[epoch_index] = KL_list
            model.local_lr = args.local_lr


        #计算平均值
        mean_intra_acc = round(np.mean(intra_accs, axis=0), 3)
        mean_inter_acc = round(np.mean(inter_accs, axis=0), 3)
        mean_intra_acc_list.append(mean_intra_acc)
        mean_inter_acc_list.append(mean_inter_acc)

        if private_dataset.setting == 'domain_skew':
            print('The ' + str(epoch_index) + ' Communcation Accuracy:' + 'Intra: ' + str(mean_intra_acc) + ' Inter: ' + str(mean_inter_acc))
        else:
            print('The ' + str(epoch_index) + ' Communcation Accuracy:' + str(mean_intra_acc))

        for i in range(len(intra_accs)):
            if i in intra_accs_dict:
                intra_accs_dict[i].append(round(intra_accs[i],2))
            else:
                intra_accs_dict[i] = [round(intra_accs[i],2)]

        for i in range(len(inter_accs)):
            if i in inter_accs_dict:
                inter_accs_dict[i].append(round(inter_accs[i],2))
            else:
                inter_accs_dict[i] = [round(inter_accs[i],2)]




    if args.csv_log:
        csv_writer.write_acc(intra_accs_dict, inter_accs_dict, mean_intra_acc_list,
                             mean_inter_acc_list,KL_loss_dict)








