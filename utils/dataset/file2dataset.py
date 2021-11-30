import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from utils.dataset.Mydataset import dataset
from utils.dataset.datasetAugFun import *
from tqdm import tqdm
from pathlib import Path


class file_dataset(object):  # 这个类主要是分数据载入场景，训练验证或是测试（可能没有标签），最后返回符合要求的数据集
    def __init__(self, data_dir, normlizetype, augment=True, stack=1):
        self.data_dir = data_dir
        self.normlizetype = normlizetype
        self.if_aug = augment
        self.num_classes = len(os.listdir(data_dir))
        self.all_npy = list(path for path in Path(self.data_dir).rglob('*.npy'))
        self.shape_len = len(np.load(self.all_npy[0])[0].shape)
        shape = np.load(self.all_npy[0])[0].shape# 某一个样本的形状
        if len(shape) >= 3:
            self.inputchannel = shape[0]
            stack = 0
        else:
            self.inputchannel = stack
        self.stack = stack
        # print(self.inputchannel)

    def dataset_marked(self, root, test=False):  # 拼接同一故障文件加下的所有文件，最后形成
        fault_type = os.listdir(root)  # 读取'E:\dataset\demo2\train'，得到故障列表
        data = []
        lab = []
        if test:
            label = 1  # 因为都在一个文件夹下，就瞎打个标‘1’
            for ii in range(len(fault_type)):  # 读取该文件夹下所有文件（测试文件应该混到一起了）
                path = os.path.join(root, fault_type[ii])
                data1, lab1 = self.data_load(path, label, self.stack)  # 返回相应长度的
                data.append(data1)
                lab.append(lab1)
        else:
            label = [i for i in range(len(fault_type))]  # 故障标签
            # print(label)
            for i in tqdm(range(len(fault_type))):  # 遍历所有故障文件夹
                npy_files = os.listdir(os.path.join(root, fault_type[i]))
                for ii in range(len(npy_files)):  # 读取该文件夹下所有文件（目前只有2个npy）
                    path = os.path.join(root, fault_type[i], npy_files[ii])
                    data1, lab1 = self.data_load(path, label[i], self.stack)
                    data.append(data1)
                    lab.append(lab1)
        list_data = [np.concatenate(data, axis=0), np.concatenate(lab, axis=0)]
        contain_nan = (True in np.isnan(list_data[0]))
        if contain_nan:  # 判断数据里是否有空值
            print('数据集中含有空值')
        # assert not contain_nan
        return list_data

    def data_load(self, filename, label, stack):  # 读取npy文件并打标
        data = np.load(filename)
        lab = np.ones(data.shape[0]) * label
        # data = np.expand_dims(data, 1)
        if stack:
            data = np.stack((data,) * stack, axis=1)
        return data, lab

    def data_transforms(self, augment, normlize_type, inputshapelen, dataset_type):
        if augment:
            if inputshapelen == 1:
                transforms = {
                    'train': Compose([
                        Normalize(normlize_type),
                        RandomAddGaussian(),
                        RandomScale(),
                        RandomStretch(),
                        RandomCrop(),
                        Retype()
                    ]),
                    'val': Compose([
                        Normalize(normlize_type),
                        Retype()
                    ])
                }
            # elif inputshapelen == 2:
            #     transforms = {
            #         'train': Compose([
            #             Normalize(normlize_type),
            #             RandomScale2d(),
            #             RandomCrop2d(),
            #             Retype()
            #         ]),
            #         'val': Compose([
            #             Normalize(normlize_type),
            #             Retype()
            #         ])
            #     }
            else:
                transforms = {
                    'train': Compose([
                        Normalize(normlize_type),
                        RandomScale2d(),
                        RandomCrop2d(),
                        Retype()
                    ]),
                    'val': Compose([
                        Normalize(normlize_type),
                        Retype()
                    ])
                }
        else:
            transforms = {
                'train': Compose([
                    Normalize(normlize_type),
                    Retype()
                ]),
                'val': Compose([
                    Normalize(normlize_type),
                    Retype()
                ])
            }
        return transforms[dataset_type]

    def data_preprare(self, val_size=0.2, random_state=0, test=False, dataset_type='val'):  # 调用之前的函数，返回数据集类dataset
        list_data = self.dataset_marked(self.data_dir, test)  # , test这里的返回的列表是本次训练的所有数[数据,标签]
        if test:
            test_dataset = dataset(list_data=list_data,
                                   transform=self.data_transforms(augment=self.if_aug, inputshapelen=self.shape_len,
                                                                  dataset_type='val', normlize_type=self.normlizetype))
            return test_dataset, '测试模式无标签'
        else:
            if val_size == 0:  # 该功能用来纯验证或纯训练
                train_dataset = dataset(list_data=list_data,
                                        transform=self.data_transforms(augment=self.if_aug,
                                                                       inputshapelen=self.shape_len,
                                                                       dataset_type=dataset_type,
                                                                       normlize_type=self.normlizetype))  # 传入的数据维numpy数组
                return train_dataset, dataset([None, None])
            else:
                train_data, val_data, train_label, val_label = train_test_split(list_data[0], list_data[1],
                                                                                test_size=val_size,
                                                                                random_state=random_state,
                                                                                stratify=list_data[1])
                # print(type(train_data),train_data.shape, val_data.shape, train_label.shape, val_label.shape)
                train_dataset = dataset(list_data=[train_data, train_label],
                                        transform=self.data_transforms(augment=self.if_aug,
                                                                       inputshapelen=self.shape_len,
                                                                       dataset_type='train',
                                                                       normlize_type=self.normlizetype))  # 传入的数据维numpy数组
                val_dataset = dataset(list_data=[val_data, val_label],
                                      transform=self.data_transforms(augment=self.if_aug, inputshapelen=self.shape_len,
                                                                     dataset_type='val',
                                                                     normlize_type=self.normlizetype))
                return train_dataset, val_dataset


if __name__ == "__main__":
    # all_set = file_dataset('E:\\dataset\\demo2\\train\\apex_drop','mean-std')
    # train_dataset, val_dataset = all_set.data_preprare(test=True)
    # print(train_dataset,len(train_dataset),val_dataset)

    all_set = file_dataset('E:\\dataset\\demo1stft\\test', '0-1', augment=True, stack=1)
    train_dataset, val_dataset = all_set.data_preprare(val_size=0, dataset_type='val')
    val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=False)
    import matplotlib.pyplot as plt
    a, _ = next(iter(val_loader))
    # for i, (data, label) in enumerate(val_loader):
    #     a = data[0].numpy().transpose((1, 2, 0))#data[0].permute((1, 2, 0))
    #     plt.imshow(a)#[0], cmap=plt.cm.gray
    #     if i % 7 == 0:
    #         plt.show()
    #         print(i, '\t', data.shape, label)
    #     break
    # print(train_dataset,len(train_dataset), val_dataset, len(val_dataset))
