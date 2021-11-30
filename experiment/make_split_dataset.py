import pandas as pd
import numpy as np
import os
from pathlib import Path
import random
from scipy.signal import resample


def old_load(path):
    np.load.__defaults__=(None, True, True, 'ASCII')
    data = np.load(path)
    np.load.__defaults__=(None, False, True, 'ASCII')
    return data


def read_file(path,mode):#返回该类别内信号组成的数组
    l = []
    if path == os.path.join('E:\dataset\demo1',mode,'basic'):##basic中通道数为8,其它为16
        #print('in basic')
        pass
    # if path == os.path.join('E:\dataset\demo1\\test\\basic'):#'E:\dataset\demo1\\test\\basic'内无文件
    #     return l
    for i in os.listdir(path):
        pp = os.path.join(path, i)
        #载入该类别npz
        data = old_load(pp)
        #取出最后一个通道
        x = data[data.files[0]][0:450560,-1]
        #print(x.shape)
        l.append(x)
    return np.asarray(l).T


def make_dataset(mode,root_path,root_path2):
    for i in ['apex_drop','basic','edge_cut','girder_defects','ice_45.4m','ice_55.4m','ice_67.4m','ice_apex']:
        print('\n'+'preprocessing '+ i)
        path = os.path.join(root_path, mode, i)
        signals = read_file(path,mode)
        signals = np.array(signals, dtype=np.float64)
        # Mean normalize X
        signals = (signals - signals.mean(axis=0)) / signals.std(axis=0)
        if not os.path.exists(os.path.join(root_path2, mode, i)):
            os.makedirs(os.path.join(root_path2, mode, i))
        for index,s in enumerate(signals.T):#该类别中的一行信号
            print('正在处理' + i + '中' + '第' + str(index) + '组数据')
            t, start, end, step = 0, 0, 4096, 4096
            while end <= len(s):
                save_name = Path(root_path2).joinpath(mode, i, i + '_' + str(index * 110 + t) + '.npy')
                np.save(save_name, s[start:end])
                #print(len(s))
                start += step
                end += step
                t += 1
                print(t,end='\t')
                #break
            #break
        #break
 
 
def make_dataset2(mode, root_path, root_path2, norm=False):
    for i in ['apex_drop','basic','edge_cut','girder_defects','ice_45.4m','ice_55.4m','ice_67.4m','ice_apex']:
        print('\n'+'preprocessing '+ i)
        path = os.path.join(root_path, mode, i)
        signals = read_file(path, mode)
        signals = np.array(signals, dtype=np.float64)
        # Mean normalize X
        if norm:
            signals = (signals - signals.mean(axis=0)) / signals.std(axis=0)# 整体标准化
        if not os.path.exists(os.path.join(root_path2, mode, i)):
            os.makedirs(os.path.join(root_path2, mode, i))
        for index,s in enumerate(signals.T):#该类别中的一行信号
            print('正在处理' + i + '中第' + str(index) + '组数据')
            t, start, end, step = 0, 0, 4096, 4096   #这里可以改切分的步长
            temp = []
            while end <= len(s):
                #save_name = Path(root_path2).joinpath(mode, i, i + '_' + str(index * 110 + t) + '.npy')
                #np.save(save_name, s[start:end])
                temp.append(s[start:end])
                start += step
                end += step
                t += 1
                print(t,end='\t')
                #break
            saved = np.asarray(temp)
            save_name = Path(root_path2).joinpath(mode, i, i + '.npy')
            np.save(save_name, saved)
            #break
        #break


class_list = ['apex_drop','basic','edge_cut','girder_defects','ice_45.4m','ice_55.4m','ice_67.4m','ice_apex']
class2label = {'apex_drop':0,'basic':1,'edge_cut':2,'girder_defects':3,'ice_45.4m':4,'ice_55.4m':5,'ice_67.4m':6,'ice_apex':7}

if __name__ == '__main__':
    root_path = 'E:\dataset\demo1'
    root_path2 = 'E:\dataset\demo2'
    for mode in [ 'train', 'test']:
        make_dataset2(mode, root_path, root_path2)#train or test,源数据地址，处理后目标地址

    # path = os.path.join(root_path, mode, class_list[3])
    # signals = read_file(path)
    # signals = np.array(signals, dtype=np.float64)
    # # Mean normalize X
    # signals = (signals - signals.mean(axis=0)) / signals.std(axis=0)
    # for s in signals.T:
    #     start, end = 0, 4096
    #     while end <= s.shape[0]:
    #         data.append(s[start:end])
    #         start += signal_size
    #         end += signal_size
    #     print(s.shape,type(s))
