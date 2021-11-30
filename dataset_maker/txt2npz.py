import pandas as pd
import numpy as np
from pathlib import Path
import os, sys


def old_load(path):#老版本的npz需要这种方式打开
    np.load.__defaults__=(None, True, True, 'ASCII')
    data = np.load(path)
    np.load.__defaults__=(None, False, True, 'ASCII')
    return data


def str2tim(a1):#没啥用，pd.Timestamp(x)可以直接读取该文本格式
    return int(time.mktime(time.strptime(a1, "T%Y-%m-%d %H:%M:%S")))


def fd2npz(basic_path):
    dirs = os.listdir(basic_path)  # 该文件下的文件目录，一种故障可能测试了多次
    datas = pd.read_csv(basic_path + 'CH07-2019-10-31 040000.txt', header=None, sep='\t').iloc[453300:453300 * 2,
            0]  # 随便找一个txt，准备读取第一列
    # print(len(datas))
    datas = datas.map(lambda x: pd.Timestamp(x))  # 读取时间戳，保存为第一列
    
    for i in dirs:  # 该故障下，读取每一个文件，并取出对应传感器的信号
        # print(i,type(i))
        data = pd.read_csv(basic_path + i, header=None, sep='\t')  # pd.read_csv
        pick = data.iloc[453300:453300 * 2, 1]
        datas = pd.concat([datas, pick], axis=1)  # 接在时间列上
        # break
    return datas.values  # 转为npy


# 打开文件
# dirs = os.listdir(root_path+'\\train')
# for file in dirs:
#    class_list.append(str(file))
# data = old_load('apex_drop1.npz')
# value_list = data.files
# x = data[value_list[0]]
class_list = ['apex_drop','basic','edge_cut','girder_defects','ice_45.4m','ice_55.4m','ice_67.4m','ice_apex']#故障文件名

if __name__ == '__main__':
    basic_path = 'E:\\dataset\\blade_turbine\\basic\\'#之后会把这个文件夹下的txt导成npz
    to_path = 'E:\\dataset\\demo1\\test\\basic\\'#demo1用来存储导出后的npz
    datas = fd2npz(basic_path)
    np.savez_compressed(Path(to_path).joinpath('basic.npz'), datas)#压缩保存，可保存多个npy数组