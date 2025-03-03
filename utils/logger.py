import copy
import os
import csv

import numpy as np

from utils.conf import base_path
from utils.util import create_if_not_exists
useless_args = ['structure','dataset', 'model', 'beta', 'csv_log', 'device_id',
                'seed', 'tensorboard','conf_jobnum','conf_timestamp','conf_host']
import pickle



class CsvWriter:
    def __init__(self, args, private_dataset):
        self.args = args
        self.private_dataset = private_dataset
        self.model_folder_path = self._model_folder_path()
        self.para_foloder_path = self._write_args()
        print(self.para_foloder_path)
    def _model_folder_path(self):
        args = self.args
        path = base_path() + args.dataset
        create_if_not_exists(path)

        if self.private_dataset.setting=='label_skew':
            path = path+'/'+str(args.beta)
            create_if_not_exists(path)

        structure_path = path+'/'+args.structure
        create_if_not_exists(structure_path)

        model_path = structure_path+'/'+args.model
        create_if_not_exists(model_path)
        return model_path


    def write_acc(self, intra_accs_dict, inter_accs_dict, mean_intra_acc_list, mean_inter_acc_list, KL_loss_dict):
        if self.private_dataset.setting == 'domain_skew':
            intra_accs_path = os.path.join(self.para_foloder_path, 'intra_accs.csv')
            self._write_all_acc(intra_accs_path,intra_accs_dict)

            inter_accs_path = os.path.join(self.para_foloder_path, 'inter_accs.csv')
            self._write_all_acc(inter_accs_path,inter_accs_dict)

            self.write_loss(KL_loss_dict,"kl_loss")

        mean_intra_path = os.path.join(self.para_foloder_path, 'mean_intra.csv')
        self._write_mean_acc(mean_intra_path,mean_intra_acc_list)

        mean_inter_path = os.path.join(self.para_foloder_path, 'mean_inter.csv')
        self._write_mean_acc(mean_inter_path,mean_inter_acc_list)


        ID=str(self.args.model_setting-1)
        mean_all_path = os.path.join(self.para_foloder_path, 'M'+ID+'.csv')
        self._write_result_acc(mean_all_path, mean_inter_acc_list,inter_accs_dict)


    def _write_args(self) -> None:
        #排除一些不必记录的参数
        args = copy.deepcopy((self.args))
        args = vars(args)
        for cc in useless_args:
            if cc in args:
                del args[cc]

        for key, value in args.items():
            args[key] = str(value)

        paragroup_dirs = os.listdir(self.model_folder_path)#获取客户端记录数据子文件夹
        n_para = len(paragroup_dirs)
        exist_para = False

        for para in paragroup_dirs:
            dict_from_csv = {}
            key_value_list = []
            para_path = os.path.join(self.model_folder_path, para)
            args_path = para_path+'/args.csv'
            with open(args_path, mode='r') as inp:
                reader = csv.reader(inp)
                for rows in reader:
                    key_value_list.append(rows)
            for index,_ in enumerate(key_value_list[0]):
                dict_from_csv[key_value_list[0][index]]=key_value_list[2][index]
            if args == dict_from_csv:
                path = para_path
                exist_para = True
                break
        if exist_para==False:
            path = os.path.join(self.model_folder_path, 'para' + str(n_para + 1))
            k=1
            while os.path.exists(path):
                path = os.path.join(self.model_folder_path, 'para' + str(n_para +k))
                k = k+1
            create_if_not_exists(path)

            columns = list(args.keys())

            write_headers = True
            args_path = path+'/args.csv'
            with open(args_path, 'a') as tmp:
                writer = csv.DictWriter(tmp, fieldnames=columns)
                if write_headers:
                    writer.writeheader()
                writer.writerow(args)
        return path
    def _write_mean_acc(self,mean_path,mean_acc_list):
        if os.path.exists(mean_path):
            with open(mean_path, 'a') as result_file:
                max_acc = max(mean_acc_list)
                max_acc_index = mean_acc_list.index(max_acc)
                for i in range(len(mean_acc_list)):
                    result = mean_acc_list[i]
                    result_file.write(str(result))
                    if i != self.args.communication_epoch - 1:
                        result_file.write(',')
                    elif (i+1)==self.args.communication_epoch:
                        result_file.write(',')
                        result_file.write(str(max_acc)+"("+str(max_acc_index)+")")
                        result_file.write('\n')
        else:
            with open(mean_path, 'w') as result_file:
                for epoch in range(self.args.communication_epoch):
                    result_file.write('epoch_' + str(epoch))
                    if epoch != self.args.communication_epoch - 1:
                        result_file.write(',')
                    elif (epoch + 1) == self.args.communication_epoch:
                        result_file.write(',')
                        result_file.write('best_epoch')
                        result_file.write('\n')
                max_acc = max(mean_acc_list)
                max_acc_index = mean_acc_list.index(max_acc)
                for i in range(len(mean_acc_list)):
                    result = mean_acc_list[i]
                    result_file.write(str(result))
                    if i != self.args.communication_epoch - 1:
                        result_file.write(',')
                    elif (i + 1) == self.args.communication_epoch:
                        result_file.write(',')
                        result_file.write(str(max_acc)+"("+str(max_acc_index)+")")
                        result_file.write('\n')

    def _write_all_acc(self,all_path,all_acc_list):
        if os.path.exists(all_path):
            with open(all_path, 'a') as result_file:
                for key in all_acc_list:
                    method_result = all_acc_list[key]
                    for epoch in range(len(method_result)):
                        result_file.write(str(method_result[epoch]))
                        if epoch != len(method_result) - 1:
                            result_file.write(',')
                        else:
                            result_file.write('\n')
        else:
            with open(all_path, 'w') as result_file:
                for epoch in range(self.args.communication_epoch):
                    result_file.write('epoch_' + str(epoch))
                    if epoch != self.args.communication_epoch - 1:
                        result_file.write(',')
                    else:
                        result_file.write('\n')

                for key in all_acc_list:
                    method_result = all_acc_list[key]
                    for epoch in range(len(method_result)):
                        result_file.write(str(method_result[epoch]))
                        if epoch != len(method_result) - 1:
                            result_file.write(',')
                        else:
                            result_file.write('\n')

    def write_loss(self, loss_dict,loss_name):
        file_path = os.path.join(self.para_foloder_path, loss_name+'.txt')
        # 使用 'a' 模式打开文件以追加内容
        with open(file_path, 'a') as f:
            # 依次写入字典中的每个列表，每个列表占一行
            for key, value in loss_dict.items():
                # 将列表内容转换为字符串，并以空格分隔
                if isinstance(value, (list, np.ndarray)):
                    # 如果值是列表或数组，将其转换为字符串并以空格分隔
                    line = ' '.join(map(str, value))
                else:
                    # 如果值是单个数值或其他类型，直接转换为字符串
                    line = str(value)
                f.write(line + '\n')

    def _write_result_acc(self, mean_dg_path, mean_inter_acc_list,inter_accs_dict):
        if os.path.exists(mean_dg_path):
            with open(mean_dg_path, 'a') as result_file:
                max_acc = max(mean_inter_acc_list)
                max_acc_index = mean_inter_acc_list.index(max_acc)
                for key in range(self.args.parti_num):
                    result_list = inter_accs_dict[key]
                    result_client = result_list[max_acc_index]
                    result_file.write(str(result_client))
                    if key != self.args.parti_num - 1:
                        result_file.write(',')
                    else:
                        result_file.write('\n')

        else:
            with open(mean_dg_path, 'w') as result_file:
                for id in range(self.args.parti_num):
                    result_file.write('Client_' + str(id+1))
                    if id != self.args.parti_num - 1:
                        result_file.write(',')
                    else :
                        result_file.write('\n')
                max_acc = max(mean_inter_acc_list)
                max_acc_index = mean_inter_acc_list.index(max_acc)
                for key in range(self.args.parti_num):
                    result_list = inter_accs_dict[key]
                    result_client = result_list[max_acc_index]
                    result_file.write(str(result_client))
                    if key != self.args.parti_num - 1:
                        result_file.write(',')
                    else:
                        result_file.write('\n')