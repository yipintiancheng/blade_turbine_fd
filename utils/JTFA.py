import scipy.signal as signal
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import scipy
import pywt
from scipy import interpolate


def plot_timedom(signal):  # 原始信号的时域和频域图
    plt.subplot(2, 1, 1)
    t = np.linspace(0, len(signal) * 0.004, len(signal))  # 定义时间轴尺度
    plt.plot(t, signal, linewidth=0.5)
    plt.title('time domain')
    plt.subplot(2, 1, 2)
    y = np.fft.fft(signal)
    # print(len(y)==len(signal), len(y), len(signal))# True 4096 4096
    f = np.linspace(0, 0.5 / 0.004, len(signal) // 2)  # 定义频率轴尺度，最高分析频率范围
    plt.plot(f, np.absolute(y)[0:len(y) // 2] * 2 / len(y), linewidth=0.5)  # 计算幅值谱，并切片出出现频率混叠之前的分量
    plt.ylim(0, np.mean(np.absolute(y)[1:len(y) // 2] * 8 / len(y)))  #
    plt.title('frequency domain')
    # plt.show()


def time2stft(x, fs=250, win_w=256, setp=60, vmax=None, boundary='zeros', window='hamming', **params):
    f, t, zxx = signal.stft(x, fs=fs, nperseg=win_w, noverlap=win_w - setp, boundary=boundary, window=window, **params)
    r = np.abs(zxx)
    # print(r.shape)  # 129*129
    # r = np.log2(r)
    # r = np.sqrt(r)
    if vmax == None:
        vmax = np.mean(r[10:-5]) * 0.002  # 定义一个极限阈值
    r[r > vmax] = vmax
    return f, t, r, vmax


def stft_specgram(x, picname=None):  # picname是给图像的名字，为了保存图像
    f, t, zxx = time2stft(x)  # np.abs(zxx).shape=129*129
    plt.pcolormesh(t, f, zxx, shading='auto', cmap='rainbow')  #
    # plt.colorbar()
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.tight_layout()
    if picname is not None:
        plt.savefig('..\\picture\\' + str(picname) + '.jpg')  # 保存图像
    return t, f, zxx


def time2dwt(aa):
    wavename = 'db5'
    cA, cD = pywt.dwt(aa, wavename)
    print(len(cA), len(cD))
    ya = pywt.idwt(cA, None, wavename, 'smooth')  # approximated component
    yd = pywt.idwt(None, cD, wavename, 'smooth')  # detailed component
    print(len(ya), len(yd))
    x = range(len(aa))
    plt.figure(figsize=(12, 9))
    plt.subplot(311)
    plt.plot(x, aa)
    plt.title('original signal')
    plt.subplot(312)
    plt.plot(x, ya)
    plt.title('approximated component')
    plt.subplot(313)
    plt.plot(x, yd)
    plt.title('detailed component')
    plt.tight_layout()
    plt.show()


def time2cwt(aa, show=False):
    sampling_rate = 256
    wavename = 'gaus8'  # 'cgau8'
    totalscal = 512  # 这个决定频率分辨率

    # 中心频率
    fc = pywt.central_frequency(wavename)
    # 计算对应频率的小波尺度
    cparam = 2 * fc * totalscal
    scales = cparam / np.arange(totalscal, 1, -1)

    [cwtmatr, frequencies] = pywt.cwt(aa, scales, wavename, 1.0 / sampling_rate)

    cwt = abs(cwtmatr)

    vmax = np.mean(cwt[:-5]) * 4 * 0.001  # 定义一个极限阈值,这个或许得手调，如果不要这个阈值的话整个图太暗了，能量全在低频部分
    cwt[cwt > vmax] = vmax

    if show:
        t = np.linspace(0, len(aa) * 0.004, len(aa))
        plt.contourf(t, frequencies, cwt)
        plt.ylabel(u"freq(Hz)")
        plt.xlabel(u"time(s)")
    return cwt, frequencies


def cwt_specgram(aa):  # 同时显示时域图和小波图
    sampling_rate = 250
    wavename = 'cgau8'  # 'cgau8'
    totalscal = 250  # 这个决定频率分辨率
    # 中心频率
    fc = pywt.central_frequency(wavename)
    print(fc)
    # 计算对应频率的小波尺度
    cparam = 2 * fc * totalscal
    scales = cparam / np.arange(totalscal, 1, -1)
    print(scales.shape)
    [cwtmatr, frequencies] = pywt.cwt(aa, scales, wavename, 1.0 / sampling_rate)
    cwt = abs(cwtmatr)
    vmax = np.mean(cwt[:-5]) * 0.005  # 定义一个极限阈值
    cwt[cwt > vmax] = vmax
    plt.figure(figsize=(8, 4))
    plt.subplot(211)
    t = np.linspace(0, len(aa) * 0.004, len(aa))
    plt.plot(t, aa)
    plt.xlabel(u"time(s)")
    plt.subplot(212)
    plt.contourf(t, frequencies, cwt)
    plt.ylabel(u"freq(Hz)")
    plt.xlabel(u"time(s)")
    plt.subplots_adjust(hspace=0.4)
    plt.show()
    return cwt, frequencies


def interp2d(arr_2d):  # 这个可以把长方形插值成正方形，放大倍数是小数进一，会损失一部分信息。。。
    transform = False
    if arr_2d.shape[1] > arr_2d.shape[0]:
        arr_2d = arr_2d.T
        transform = True
    rows, cols = arr_2d.shape
    repeats = round(rows // cols)
    arr_2d = arr_2d.repeat(repeats).reshape((rows, -1))
    arr_2d = arr_2d[:, :rows]
    if transform:
        arr_2d = arr_2d.T
    return arr_2d


def interp2d2(arr, shape=None, kind='linear'):  # *优点是简单易用，可以实现不限于变成正方形等操作，缺点是没有我想要的零阶插值
    if shape is None:
        h = w = max(arr.shape)
    else:
        (h, w) = shape
    fun = interpolate.interp2d(np.linspace(0, 100, num=arr.shape[1]), np.linspace(0, 100, num=arr.shape[0]), arr, kind=kind)
    arr = fun(np.linspace(0, 100, num=w), np.linspace(0, 100, num=h))
    return arr


def interp2d3(arr, shape):  # **最近邻插值算法
    (dstH, dstW) = shape
    scrH, scrW = arr.shape
    result = np.zeros((dstH, dstW))  # , dtype=np.uint8
    for i in range(dstH - 1):
        for j in range(dstW - 1):
            scrx = round(i * (scrH / dstH))
            scry = round(j * (scrW / dstW))
            result[i, j] = arr[scrx, scry]
    return result


def interp3d(arr_3d):
    """三维数组线性插值

    arr_3d      - numpyp.ndarray类型的三维数组
    """

    layers, rows, cols = arr_3d.shape

    arr_3d = arr_3d.repeat(2).reshape((layers * rows, -1))
    arr_3d = arr_3d.repeat(2, axis=0).reshape((layers, -1))
    arr_3d = arr_3d.repeat(2, axis=0).reshape((layers * 2, rows * 2, cols * 2))[:-1, :-1, :-1]

    arr_3d[:, :, 1::2] += (arr_3d[:, :, 2::2] - arr_3d[:, :, 1::2]) / 2
    arr_3d[:, 1::2, :] += (arr_3d[:, 2::2, :] - arr_3d[:, 1::2, :]) / 2
    arr_3d[1::2, :, :] += (arr_3d[2::2, :, :] - arr_3d[1::2, :, :]) / 2

    return arr_3d


if __name__ == '__main__':
    def p5(path, loc):
        l = list(path for path in Path(path).rglob('*.npy'))  # 一个类别一个npy文件
        plt.figure(figsize=(len(l) * 5, 1 * 5))
        plt.title(path)
        for i, p in enumerate(l):
            plt.subplot(1, len(l), i + 1)
            s = np.load(p)[loc]
            stft_specgram(s)  # 绘制短时傅里叶图
            # time2cwt(s,True)# 绘制cwt时频图
        plt.show()


    # p5('E:\\dataset\demo2\\test', 5)
    # p5('E:\\dataset\demo2\\train', 5)
    path = 'E:\\dataset\demo2\\test'
    l = list(path for path in Path(path).rglob('*.npy'))
    s = np.load(l[1])[5]
    # t, f, r = time2stft(s)
    x = s
    _, _, x0, v = time2stft(x, win_w=256, setp=32)
    a = x0.shape
    # _, _, x1, _ = time2stft(x, win_w=128, setp=32, vmax=v)
    # x1 = interp2d3(x1, a)
    _, _, x2, _ = time2stft(x, win_w=128, setp=64, vmax=v)
    x2 = interp2d3(x2, a)
    _, _, x3, _ = time2stft(x, win_w=64, setp=64, vmax=v)
    x3 = interp2d3(x3, a)
    _, _, x4, _ = time2stft(x, win_w=256, setp=128, vmax=v)
    x4 = interp2d3(x4, a)
    ll = [x4, x0, x2, x3]
    plt.figure(figsize=(len(ll) * 6, 1 * 4))
    for i, p in enumerate(ll):
        ax = plt.subplot(1, len(ll), i + 1)
        # plt.title(p.parent.name+' frequency spectrum')
        signal = p  #
        ax.pcolormesh(signal, shading='auto', cmap='rainbow')
    plt.show()
    # x = np.asarray([x3, x0, x1, x2])
    # print(x.shape)
    # cwtmatr, frequencies = time2cwt(s, show=False)
    # a = cwtmatr[:, 64:100]
