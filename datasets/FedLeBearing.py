import os
from argparse import ArgumentParser

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils import data as da
import scipy.io as scio
from torch.utils.data import SubsetRandomSampler


class Dataset(da.Dataset):
    def __init__(self,x,y,z):
        self.Data =x
        self.Label = y
        self.domain_labels=z
    def __getitem__(self, index):
        txt = self.Data[index]
        label = self.Label[index]
        domain_labels=self.domain_labels[index]
        return txt, label,domain_labels
    def __len__(self):
        return len(self.Data)

class Dataset_public(da.Dataset):
    def __init__(self,x,y,z):
        self.Data =x
        self.Label = y
        self.domain_labels = z

    def __getitem__(self, index):
        txt = self.Data[index]
        label = self.Label[index]
        domain_labels=self.domain_labels[index]
        return txt, label,domain_labels
    def __len__(self):
        return len(self.Data)

def capture_mat_all(original_path,domain="",dataset="",class_n=0):
        file_name_list = os.listdir(original_path+"/"+domain)
        for i in file_name_list:
            file_path=os.path.join(original_path+"/"+domain,i)
            data_temp = scio.loadmat(file_path)
            if dataset in ["bearing-sdust","bearing-hust","bearing-sdust","bearing-pu","CWRU5","CWRU7","CWRU9","CWRU10","PU-Bearing"]:
                data = data_temp.get("BearingData_"+domain+"_"+str(class_n)+"_Norm")
            else:
                data = data_temp.get("GearData_" + domain + "_" + str(class_n) + "_Norm")
            # 读取样本特征数据和标签数据
            data_x = data[:, :-1]
            data_y = data[:, -1]


        return data_x,data_y

class FedLeaBearing_priv():

    def __init__(self,args=None):
        self.args = args
        self.public_name = self.args.public_dataset
        self.batch_size = self.args.public_batch_size
        self.dataset = args.dataset
        self.setting  = args.setting
        self.imbalance_dict={}
        if self.setting == 'domain_skew' and self.dataset=="bearing-hust":
            self.DOMAINS_LIST = ['60','65', '70', '75', '80']
            self.percent_dict = {'60': 0.5,'65': 0.5, '70': 0.5, '75': 0.5, '80': 0.5}#测试集比例
            self.N_CLASS = 9
        elif self.setting == 'domain_skew' and self.dataset=="gearbox-bjut":
            self.DOMAINS_LIST = ['1200', '1800', '2400', '3000']
            self.percent_dict = {'1200': 0.5, '1800': 0.5, '2400': 0.5, '3000':0.5}#测试集比例
            self.N_CLASS = 5
        elif self.setting == 'domain_skew'and self.dataset=="gearbox-sdust":
            self.DOMAINS_LIST = ['1000', '1500', '2000', '2500']
            self.percent_dict = {'1000': 0.5, '1500': 0.5, '2000': 0.5, '2500': 0.5}
            self.N_CLASS = 7
        elif self.setting == 'domain_skew'and self.dataset=="bearing-sdust":
            self.DOMAINS_LIST = ['1000', '1500', '2000', '2500']
            self.percent_dict = {'1000': 0.5, '1500': 0.5, '2000': 0.5, '2500': 0.5}
            self.N_CLASS = 10
        elif self.setting == 'domain_skew'and self.dataset=="bearing-pu":
            self.DOMAINS_LIST = ['971', '1511', '1571', '1574']
            self.percent_dict = {'971': 0.5, '1511': 0.5, '1571': 0.5, '1574': 0.5}
            self.N_CLASS = 5

    def prepro_data(self,file_path,dataset="", domain='0',test_rate=0.2, class_n=10):
            # 提取数据
            domain_speed = domain
            data_x, data_y = capture_mat_all(original_path=file_path, domain=domain_speed, dataset=dataset,class_n=class_n)
            # 划分训练集与测试集
            train_data_x, test_data_x, train_data_y, test_data_y = train_test_split(data_x, data_y, test_size=test_rate, shuffle=True)


            train_domain_labels = [int(domain_speed) for i in range(len(train_data_y))]
            test_domain_labels = [int(domain_speed)  for i in range(len(test_data_y))]

            train_domain_labels = np.array(train_domain_labels)
            test_domain_labels = np.array(test_domain_labels)
            train_data_x= torch.FloatTensor(train_data_x).unsqueeze(1)
            train_data_y = torch.LongTensor(train_data_y)
            train_dataset = Dataset(train_data_x, train_data_y,train_domain_labels)
            train_dataloader = da.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,drop_last=True)


            test_data_x = torch.FloatTensor(test_data_x).unsqueeze(1)
            test_data_y = torch.LongTensor(test_data_y)
            test_dataset = Dataset(test_data_x, test_data_y,test_domain_labels)
            test_dataloader = da.DataLoader(test_dataset, batch_size=self.batch_size)
            return train_dataloader, test_dataloader


    def get_data_loaders(self, selected_domain_list=[]):

        using_list = self.DOMAINS_LIST if selected_domain_list == [] else selected_domain_list
        train_loader_list = []
        test_loader_list = []
        #制作训练和测试集

        if self.setting == 'domain_skew':
            for _, domain in enumerate(using_list):
                test_rate=self.percent_dict[domain]
                if self.dataset=="bearing-hust":
                    train_loader,test_loader= self.prepro_data(r'./datasets/HUST-Bearing',self.dataset,domain,test_rate,class_n=self.N_CLASS)
                    train_loader_list.append(train_loader)
                    test_loader_list.append(test_loader)
                elif self.dataset=="gearbox-bjut":
                    train_loader,test_loader= self.prepro_data(r'./datasets/BJUT',self.dataset,domain,test_rate,class_n=self.N_CLASS)
                    train_loader_list.append(train_loader)
                    test_loader_list.append(test_loader)
                elif self.dataset=="gearbox-sdust":
                    train_loader, test_loader = self.prepro_data(r'./datasets/SDUST-Gearbox',self.dataset,domain,test_rate, class_n=self.N_CLASS)
                    train_loader_list.append(train_loader)
                    test_loader_list.append(test_loader)
                elif self.dataset=="bearing-sdust":
                    train_loader, test_loader = self.prepro_data(r'./datasets/SDUST-Bearing',self.dataset,domain,test_rate, class_n=self.N_CLASS)
                    train_loader_list.append(train_loader)
                    test_loader_list.append(test_loader)
                elif self.dataset=="bearing-pu":
                    train_loader, test_loader = self.prepro_data(r'./datasets/PU-Bearing',self.dataset,domain,test_rate, class_n=self.N_CLASS)
                    train_loader_list.append(train_loader)
                    test_loader_list.append(test_loader)

        return train_loader_list, test_loader_list

class FedLeaBearing_publ():
    def __init__(self,args=None,cwru_value=0):
        self.args = args
        self.cwru = cwru_value
        self.public_name = self.args.public_dataset
        self.batch_size = self.args.public_batch_size
        self.dataset = args.dataset
        if self.public_name == 'CWRU5':
            self.DOMAINS_LIST = ['1730', '1750', '1772', '1797']
            self.original_path = r'./datasets/' + self.public_name
            self.N_CLASS = 5
        elif self.public_name == 'CWRU7':
            self.DOMAINS_LIST = ['1730', '1750', '1772', '1797']
            self.original_path = r'./datasets/' + self.public_name
            self.N_CLASS = 7
        elif self.public_name == 'CWRU9':
            self.DOMAINS_LIST = ['1730', '1750', '1772', '1797']
            self.original_path = r'./datasets/' + self.public_name
            self.N_CLASS = 9
        elif self.public_name == 'CWRU10':
            self.DOMAINS_LIST = ['1730', '1750', '1772', '1797']
            self.original_path = r'./datasets/' + self.public_name
            self.N_CLASS = 10
        elif self.public_name == 'PU-Bearing':
            self.DOMAINS_LIST = ['971', '1511', '1571', '1574']
            self.original_path = r'./datasets/' + self.public_name
            self.N_CLASS = 5


    def get_data_loaders(self):
        for idx ,i in enumerate(self.DOMAINS_LIST):
            data_x,data_y=capture_mat_all(self.original_path,domain=i,dataset=self.public_name,class_n=self.N_CLASS)
            train_domain_labels = [i for j in range(len(data_y))]
            if idx==0:
                data_all_x=data_x
                data_all_y=data_y
                train_domain_all=train_domain_labels
            else:
                data_all_x=np.concatenate((data_all_x,data_x),axis=0)
                data_all_y=np.concatenate((data_all_y,data_y),axis=0)
                train_domain_all=np.concatenate((train_domain_all,train_domain_labels),axis=0)


        train_x = torch.from_numpy(np.array(data_all_x, np.float32)).unsqueeze(1)
        train_y= torch.LongTensor(data_all_y)
        train_domain_labels = torch.from_numpy(np.array(train_domain_all, np.float32))


        train_dataset = Dataset_public(train_x, train_y,train_domain_labels)
        n_train=len(train_dataset)
        idxs=np.random.permutation(n_train)
        if self.public_name=="CWRU5":
            sampler=2000
        elif self.public_name=="CWRU7":
            sampler =int(2800*self.cwru)
        elif self.public_name=="CWRU9":
            sampler = int(3600*self.cwru)
        elif self.public_name=="CWRU10":
            sampler = 4000
        elif self.public_name=="PU-Bearing":
            sampler = 2000
        idxs = idxs[0:sampler]
        train_sampler = SubsetRandomSampler(idxs)
        train_dataloader = da.DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,drop_last=True)
        return train_dataloader

