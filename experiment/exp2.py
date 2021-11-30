import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from utils.JTFA import *


# from dataset_maker.make_dataset import *
# import dataset_maker.make_dataset.class_list
# print(class_list)
# for ii in [-4,-3,-2,-1]:
#     print(ii)
# data_pd = pd.DataFrame({"data": np.zeros((10,2)).tolist(), "label": np.zeros((10,5)).tolist()})
# print(data_pd.loc[1:3,'label'])


# l = [0,0,0,0,0,0]
# ll = [1,1,1,1,1,1]
# print(l+ll)
# l=[1 for i in range(5)]
# np.ones(10)*5


# import pathlib
# from pathlib import Path
# import numpy as np
# path = Path.cwd()
#
# p_path = path.parent
# #[path for path in p_path.iterdir() if path.is_dir()]
# #[str(path) for path in Path(set).iterdir()]
# set='E:\dataset\demo2\\test'
# pppp = str(next(Path('E:\dataset\demo2\\test').rglob('*.npy')))
# #input_channal = (np.load(str(next(Path(set).rglob('*.npy')))).shape[1]  if np.load(str(next(Path(set).rglob('*.npy')))).shape[1]<=20 else 1)
# #[path for path in Path(set).iterdir()]


def plot_timedom(signal):
    plt.subplot(2, 1, 1)
    t = np.linspace(0, len(signal) * 0.004, len(signal))  # 定义时间轴尺度
    plt.plot(t, signal, linewidth=0.5)
    plt.title('time domain')
    plt.subplot(2, 1, 2)
    y = np.fft.fft(signal)
    # print(len(y)==len(signal), len(y), len(signal))# True 4096 4096
    f = np.linspace(0, 0.5 / 0.004, len(signal) // 2)  # 定义频率轴尺度，最高分析频率范围
    plt.plot(f, np.absolute(y)[0:len(y) // 2] * 2 / len(y), linewidth=0.5)  # 计算幅值谱，并切片出出现频率混叠之前的分量
    plt.title('frequency domain')
    plt.show()


pp = Path('E:\dataset\demo1stft\\test')
ll = list(path.name for path in pp.iterdir())
l = list(path for path in pp.rglob('*.npy'))
# print(l)
# data = np.load('E:\dataset\demo4\\train')
# data.shape
n = 20

class_list = ['apex_drop', 'basic', 'edge_cut', 'girder_defects', 'ice_45.4m', 'ice_55.4m', 'ice_67.4m', 'ice_apex']
plt.figure(figsize=(len(l) * 5, 1 * 5))
for i, p in enumerate(l):
    ax = plt.subplot(1, len(l), i + 1)
    # plt.title(p.parent.name+' frequency spectrum')
    signal = np.load(p)[n]  #
    ax.set_title(class_list[i])
    ax.pcolormesh(signal, shading='auto', cmap='rainbow')
    # f = np.linspace(0, 0.5 / 0.004, len(signal))
    # print(len(signal))
    # plt.ylim(0, 1e-17)
    # plt.plot(f, signal, linewidth=0.5)
# plt.savefig('fft_detrend_signal'+ str(n)+'.png')
plt.savefig('E:\dataset\demo1stft\\test_[' + str(n) + '].png')
plt.show()
