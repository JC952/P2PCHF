import os
import sys
from argparse import ArgumentParser
from random import random
import numpy as np
from datasets.FedLeBearing import FedLeaBearing_priv, FedLeaBearing_publ
from models import get_all_models, get_model
from utils.args import add_management_args
from utils.best_args import best_args
from models.ClientFL.Client_net1 import Client_net1
from models.ClientFL.Client_net2 import Client_net2
from models.ClientFL.Client_net3 import Client_net3
from models.ClientFL.Client_net4 import Client_net4
from models.ClientFL.Client_net5 import Client_net5
from models.ClientFL.Client_net6 import Client_net6
from utils.training import train
import torch
import time
import uuid
import datetime
torch.multiprocessing.set_sharing_strategy('file_system')

conf_path = os.getcwd()
sys.path.append(conf_path)
sys.path.append(conf_path + '/datasets')
sys.path.append(conf_path + '/backbone')
sys.path.append(conf_path + '/models')



def parse_args(setting_data=1,temp=0,fisl=0,fccm=0):

    parser = ArgumentParser(description='You Only Need Me', allow_abbrev=False)
    parser.add_argument('--device_id', type=int, default=0, help='The Device Id for Experiment')
    parser.add_argument('--parti_num', type=int, default=0, help='The Number for Participants')  # Domain 4 Label 10
    parser.add_argument('--model', type=str, default='p2pchf',  help='Model name.', choices=get_all_models())
    parser.add_argument('--setting', type=str, default='domain_skew')
    parser.add_argument('--structure', type=str, default='heterogeneity')
    if setting_data == 1:
        parser.add_argument('--dataset', type=str, default='gearbox-sdust', help='Private Dataset name.')
        parser.add_argument('--public_dataset', type=str, default='CWRU7')
    elif setting_data == 2:
        parser.add_argument('--dataset', type=str, default='gearbox-bjut', help='Private Dataset name.')
        parser.add_argument('--public_dataset', type=str, default='CWRU5')
    elif setting_data == 3:
        parser.add_argument('--dataset', type=str, default='bearing-hust', help='Private Dataset name.')
        parser.add_argument('--public_dataset', type=str, default='CWRU9')
    parser.add_argument('--model_setting', type=int, default=4, help='M3')
    parser.add_argument('--get_time', type=int, default=0)
    add_management_args(parser)
    args = parser.parse_args()
    best = best_args[args.dataset][args.model]
    best = best[-1]
    for key, value in best.items():
        setattr(args, key, value)

    args.temp=0.5
    args.fisl=fisl
    args.fccm = fccm
    if args.dataset=="bearing-hust":
        args.parti_num = 5
    else:
        args.parti_num = 4
    return args




def main(args=None,data=1,loop=1,temp=0,fisl=0,fccm=0,cwru=0):
    setting_data = data
    if args is None:
        args = parse_args( setting_data=setting_data,temp=temp,fisl=fisl,fccm=fccm)

    args.conf_jobnum = str(uuid.uuid4())
    args.conf_timestamp = str(datetime.datetime.now())

    # 返回私有数据集
    private_dataset = FedLeaBearing_priv(args)
    # 返回公共数据集
    public_dataset= FedLeaBearing_publ(args,cwru_value=cwru)

    #客户端模型异构，获取模型名称
    if args.structure == 'heterogeneity' and setting_data!=3:
        models_names_list= ['Clent_net1','Clent_net2','Clent_net3', 'Clent_net4',]
    else:
        models_names_list = ['Clent_net1', 'Clent_net2', 'Clent_net3', 'Clent_net4','Clent_net5']


    models_list = [ ]
    for _,name_id in enumerate(models_names_list):
        if name_id=='Clent_net1':
            models_list.append( Client_net1(private_dataset.N_CLASS))
        elif name_id=='Clent_net2':
            models_list.append( Client_net2(private_dataset.N_CLASS))
        elif name_id=='Clent_net3':
            models_list.append(Client_net3(private_dataset.N_CLASS))
        elif name_id=='Clent_net4':
            models_list.append(Client_net4(private_dataset.N_CLASS))
        elif name_id=='Clent_net5':
            models_list.append(Client_net5(private_dataset.N_CLASS))
        elif name_id=='Clent_net6':
            models_list.append(Client_net6(private_dataset.N_CLASS))




    model = get_model(models_list, args)
    if args.structure not in model.COMPATIBILITY:
        print(model.NAME + ' does not support model heterogeneity')
    else:
        print('{}_{}_{}_{}_{}_{}_{}'.format(args.model,args.structure, args.dataset,args.setting,
                                            args.communication_epoch, args.public_dataset,args.local_epoch))
    method=args.model_setting
    data_name=setting_data
    train(model, public_dataset, private_dataset, args,loop=loop,method=method,data_name=data_name)#模型训练
#

if __name__ == '__main__':
    start_time = time.time()
    fisl=[1.5]
    fccm=[0.005]
    for fisl_value in fisl:
        for fccm_value in fccm:
            for a in [2]:
                for i in range(5):
                    main(data=a,loop=i,fisl=fisl_value,fccm=fccm_value)

    end_time = time.time()
    training_time = end_time - start_time
    print("训练时间（秒）:", training_time)


