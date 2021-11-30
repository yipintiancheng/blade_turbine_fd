import pandas as pd
import numpy as np
import os, sys

def old_load(path):
    np.load.__defaults__=(None, True, True, 'ASCII')
    data = np.load(path)
    np.load.__defaults__=(None, False, True, 'ASCII')
    return data

root_path = 'E:\dataset\demo3'
class_list = ['apex_drop','basic','edge_cut','girder_defects','ice_45.4m','ice_55.4m','ice_67.4m','ice_apex',]
#x = np.load('E:\dataset\demo1\\train\\basic\\basic.npz')

data1 = old_load('E:\\dataset\\demo1\\train\\ice_55.4m\\ice_55.4m_60kg_1.npz')#这里面好多空值
value_list1 = data1.files
x1 = data1[value_list1[0]]
print(x1.shape)
data2 = old_load('E:\dataset\demo1\\train\edge_cut\\edge_cut1_1.npz')
value_list2 = data2.files
x2 = data2[value_list2[0]]
print(x2.shape)









# time 与 pd.Timestamp
# import time
#
# a1 = "T2019-5-10 23:40:00"
#
# # 先转换为时间数组
# timeArray = time.strptime(a1, "T"+"%Y-%m-%d %H:%M:%S")
# # 转换为时间戳
# timeStamp = int(time.mktime(timeArray))
#
# datetime = pd.Timestamp(a1)
#
# print(type(datetime))#一个是字符串，一个是时间戳格式













# import os
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from utils.dataset_loader.Mydataset import dataset
# from utils.dataset_loader.datasetAugFun import *
# from tqdm import tqdm
#
# #generate Training Dataset and Testing Dataset

# def dataset_marked(root, test=False):
#     fault_type = os.listdir(root)  # 读取'E:\dataset\demo3\train'，得到故障列表
#     label = [i for i in range(len(fault_type))]#故障标签
#     # label2fault = dict(zip(label,fault_type))#标签对应的故障类型
#     # fault2label = dict(zip(fault_type,label))#故障类型对应的标签
#     data = []
#     lab = []
#     for i in tqdm(range(len(fault_type))):#遍历所有故障文件夹
#         npy_files = os.listdir(os.path.join(root, fault_type[i]))
#         for ii in range(len(npy_files)): #读取该文件夹下所有文件（目前只有一个npy）
#             path = os.path.join(root, fault_type[i], npy_files[ii])
#             data1, lab1 = data_load(path, label[i])#返回相应长度的
#             data += data1#列表和列表相加会在后面拼接起来
#             lab += lab1
#     list_data = [np.concatenate(data, axis=0),np.concatenate(lab, axis=0)]
#     return list_data
#
#
# def data_load(filename, label):#读取npy文件打标
#     data = np.load(filename)
#     lab = np.ones(data.shape[0]) * label
#     data = data
#     return [data], [lab]
#
#
#
# a = dataset_marked('E:\\dataset\\demo2\\train')
# #b,c = np.array(a[0]),np.array(a[1])
