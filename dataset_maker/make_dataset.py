import pandas as pd
import numpy as np
import os
from pathlib import Path
import random
from scipy.signal import resample
from scipy import signal
from utils.JTFA import time2stft, time2cwt, interp2d3
from utils.feaxtra import feature


def old_load(path):
    np.load.__defaults__ = (None, True, True, 'ASCII')
    data = np.load(path)
    np.load.__defaults__ = (None, False, True, 'ASCII')
    return data


def read_file(path):  # 返回该类别内信号组成的数组
    l = []
    # if path == os.path.join('E:\dataset\demo1', mode, 'basic'):  ##basic中通道数为8,其它为16
    #     # print('in basic')
    #     pass
    # # if path == os.path.join('E:\dataset\demo1\\test\\basic'):#'E:\dataset\demo1\\test\\basic'内无文件
    # #     return l
    for i in os.listdir(path):
        pp = os.path.join(path, i)
        # 载入该类别npz
        data = old_load(pp)
        # 取出最后一个通道，这里还需要调整，日后迟早要尝试多通道的，可以尝试把通道选择暴露出来（-1）
        x = data[data.files[0]][0:450560, -1]
        # print(x.shape)
        l.append(x)
    return np.asarray(l).T


def make_dataset(mode, root_path, root_path2):  # 这个输出的样本是一个一个的不太好，但是可能看着舒服
    for i in ['apex_drop', 'basic', 'edge_cut', 'girder_defects', 'ice_45.4m', 'ice_55.4m', 'ice_67.4m', 'ice_apex']:
        print('\n' + 'preprocessing ' + i)
        path = os.path.join(root_path, mode, i)
        signals = read_file(path)
        signals = np.array(signals, dtype=np.float64)
        # Mean normalize X
        signals = (signals - signals.mean(axis=0)) / signals.std(axis=0)
        if not os.path.exists(os.path.join(root_path2, mode, i)):
            os.makedirs(os.path.join(root_path2, mode, i))
        for index, s in enumerate(signals.T):  # 该类别中的一行信号
            print('正在处理' + i + '中' + '第' + str(index) + '组数据')
            t, start, end, step = 0, 0, 4096, 4096
            while end <= len(s):
                save_name = Path(root_path2).joinpath(mode, i, i + '_' + str(index * 110 + t) + '.npy')
                np.save(save_name, s[start:end])
                # print(len(s))
                start += step
                end += step
                t += 1
                print(t, end='\t')
                # break
            # break
        # break


def make_dataset2(mode, root_path, root_path2, norm=False):  # 之后的版本一个类别都会集成在一个npy里
    for i in ['apex_drop', 'basic', 'edge_cut', 'girder_defects', 'ice_45.4m', 'ice_55.4m', 'ice_67.4m', 'ice_apex']:
        print('\n' + 'preprocessing ' + i)
        path = os.path.join(root_path, mode, i)
        signals = read_file(path)
        signals = np.array(signals, dtype=np.float64)
        # Mean normalize X
        if norm:
            signals = (signals - signals.mean(axis=0)) / signals.std(axis=0)  # 整体标准化
        if not os.path.exists(os.path.join(root_path2, mode, i)):
            os.makedirs(os.path.join(root_path2, mode, i))
        for index, s in enumerate(signals.T):  # 该类别中的一行信号
            print('正在处理' + i + '中第' + str(index) + '组数据')
            t, start, end, step = 0, 0, 4096, 4096  # 这里可以改切分的步长
            temp = []
            while end <= len(s):
                # save_name = Path(root_path2).joinpath(mode, i, i + '_' + str(index * 110 + t) + '.npy')
                # np.save(save_name, s[start:end])
                temp.append(s[start:end])
                start += step
                end += step
                t += 1
                print(t, end='\t')
                # break
            saved = np.asarray(temp)
            save_name = Path(root_path2).joinpath(mode, i, i + str(index) + '.npy')
            np.save(save_name, saved)
            # break
        # break


def make_fft_dataset(mode, root_path, root_path2, norm=True, detrend=True):
    print(list(path.name for path in Path(root_path).joinpath(mode).iterdir()))
    for i in list(path.name for path in Path(root_path).joinpath(mode).iterdir()):
        print('\n' + 'preprocessing ' + i)
        path1 = os.path.join(root_path, mode, i)
        print('\n' + path1)
        signals = read_file(path1)
        signals = np.array(signals, dtype=np.float64)
        # Mean normalize X
        if norm:
            signals = (signals - signals.min(axis=0)) / (signals.max(axis=0) - signals.min(axis=0))
        if detrend:
            signals = signal.detrend(signals)
        if not os.path.exists(os.path.join(root_path2, mode, i)):
            os.makedirs(os.path.join(root_path2, mode, i))
        for index, s in enumerate(signals.T):  # 该类别中的一行信号
            print('\n正在处理' + i + '中第' + str(index) + '组数据')
            t, start, end, step = 0, 0, 4096, 4096  # 这里可以改切分的步长
            temp = []
            while end <= len(s):
                # save_name = Path(root_path2).joinpath(mode, i, i + '_' + str(index * 110 + t) + '.npy')
                # np.save(save_name, s[start:end])
                x = s[start:end]
                # x = (signal.detrend(x) if detrend else x)
                x = np.fft.fft(x)
                x = np.abs(x) * 2 / len(x)
                x = x[range(int(len(x) // 2))]
                # print(len(x))
                temp.append(x)
                start += step
                end += step
                t += 1
                print(t, end='\t')
                # break
            saved = np.asarray(temp)
            save_name = Path(root_path2).joinpath(mode, i, i + str(index) + '.npy')
            np.save(save_name, saved)
            # break
        # break


def make_stft_dataset(mode, root_path, root_path2, norm=False, detrend=False):
    print(list(path.name for path in Path(root_path).joinpath(mode).iterdir()))
    for i in list(path.name for path in Path(root_path).joinpath(mode).iterdir()):
        print('\n' + 'preprocessing ' + i)
        path1 = os.path.join(root_path, mode, i)
        print('\n' + path1)
        signals = read_file(path1)
        signals = np.array(signals, dtype=np.float64)
        print(signals.shape)
        # Mean normalize X
        if norm:
            signals = (signals - signals.min(axis=0)) / (signals.max(axis=0) - signals.min(axis=0))
            # signals = (signals - signals.mean(axis=0)) / signals.std(axis=0)  # 整体标准化
        if detrend:
            signals = signal.detrend(signals)
        if not os.path.exists(os.path.join(root_path2, mode, i)):
            os.makedirs(os.path.join(root_path2, mode, i))
        for index, s in enumerate(signals.T):  # 该类别中的一行信号
            print('\n正在处理' + i + '中第' + str(index) + '组数据')
            t, start, end, step = 0, 0, 4096, 4096  # 这里可以改切分的步长
            temp = []
            while end <= len(s):
                x = s[start:end]
                _, _, x0, v = time2stft(x, win_w=256, setp=32)
                #####
                a = x0.shape
                # _, _, x1, _ = time2stft(x, win_w=128, setp=32, vmax=v)
                # x1 = interp2d3(x1, a)
                _, _, x2, _ = time2stft(x, win_w=128, setp=64, vmax=v)
                x2 = interp2d3(x2, a)
                _, _, x3, _ = time2stft(x, win_w=64, setp=64, vmax=v)
                x3 = interp2d3(x3, a)
                _, _, x4, _ = time2stft(x, win_w=256, setp=128, vmax=v)
                x4 = interp2d3(x4, a)
                x = np.asarray([x4, x0, x2, x3])
                #####
                print(x.shape)
                temp.append(x)
                start += step
                end += step
                t += 1
                print(t, end='\t')
                # break
            saved = np.asarray(temp)
            print(saved.shape)
            save_name = Path(root_path2).joinpath(mode, i, i + str(index) + '.npy')
            np.save(save_name, saved)
            # break
        # break


def make_cwt_dataset(mode, root_path, root_path2, norm=False, detrend=False):
    print(list(path.name for path in Path(root_path).joinpath(mode).iterdir()))
    for i in list(path.name for path in Path(root_path).joinpath(mode).iterdir()):
        print('\n' + 'preprocessing ' + i)
        path1 = os.path.join(root_path, mode, i)
        print('\n' + path1)
        signals = read_file(path1)
        signals = np.array(signals, dtype=np.float64)
        print(signals.shape)
        # Mean normalize X
        if norm:
            signals = (signals - signals.min(axis=0)) / (signals.max(axis=0) - signals.min(axis=0))
            # signals = (signals - signals.mean(axis=0)) / signals.std(axis=0)  # 整体标准化
        if detrend:
            signals = signal.detrend(signals)
        if not os.path.exists(os.path.join(root_path2, mode, i)):
            os.makedirs(os.path.join(root_path2, mode, i))
        for index, s in enumerate(signals.T):  # 该类别中的一行信号
            print('\n正在处理' + i + '中第' + str(index) + '组数据')
            t, start, end, step = 0, 0, 1024, 1024  # 这里可以改切分的步长
            temp = []
            while end <= len(s):
                x = s[start:end]
                x, _ = time2cwt(x)
                x = x[:, 256:768]
                # print(x.shape)
                # import matplotlib.pyplot as plt
                # plt.imshow(x)
                # plt.show()
                temp.append(x)
                start += step
                end += step
                t += 1
                print(t, end='\t')
                # break
            saved = np.asarray(temp)
            print(saved.shape)
            save_name = Path(root_path2).joinpath(mode, i, i + str(index) + '.npy')
            np.save(save_name, saved)
            # break
        # break


def make_fea_dataset(mode, root_path, root_path2, norm=False, detrend=False):  # 时域频域特征值
    print(list(path.name for path in Path(root_path).joinpath(mode).iterdir()))
    for i in list(path.name for path in Path(root_path).joinpath(mode).iterdir()):
        print('\n' + 'preprocessing ' + i)
        path1 = os.path.join(root_path, mode, i)
        print('\n' + path1)
        signals = read_file(path1)
        signals = np.array(signals, dtype=np.float64)
        print(signals.shape)
        # Mean normalize X
        if norm:
            signals = (signals - signals.min(axis=0)) / (signals.max(axis=0) - signals.min(axis=0))
            # signals = (signals - signals.mean(axis=0)) / signals.std(axis=0)  # 整体标准化
        if detrend:
            signals = signal.detrend(signals)
        if not os.path.exists(os.path.join(root_path2, mode, i)):
            os.makedirs(os.path.join(root_path2, mode, i))
        for index, s in enumerate(signals.T):  # 该类别中的一行信号
            print('\n正在处理' + i + '中第' + str(index) + '组数据')
            t, start, end, step = 0, 0, 1024, 1024  # 这里可以改切分的步长
            temp = []
            while end <= len(s):
                x = s[start:end]
                f = feature(x)
                x = f.Both_Fea()
                temp.append(x)
                start += step
                end += step
                t += 1
                print(t, end='\t')
            saved = np.asarray(temp)
            print(saved.shape)
            save_name = Path(root_path2).joinpath(mode, i, i + str(index) + '.npy')
            np.save(save_name, saved)
            # break
        # break
    pass


if __name__ == '__main__':
    root_path = 'E:\\dataset\\npz_source'
    root_path2 = 'E:\\dataset\\demo84stft'
    for mode in ['train', 'test']:
        print('\n' + mode + '**************************')
        make_stft_dataset(mode, root_path, root_path2, norm=False, detrend=False)  # train or test,源数据地址，处理后目标地址
        # break
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
