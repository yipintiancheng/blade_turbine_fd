import numpy as np
import numpy.fft as npfft
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from pathlib import Path
import model
from utils.dataset.file2dataset import file_dataset
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix


# labels = [label_to_index[pathlib.Path(path).parent.name] for path in paths]#给paths里的文件打标

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


class tester(object):
    def __init__(self, model_path, set='E:\dataset', stack=1):  # 获取模型（权重路径和模型类名）和数据集（路径）
        self.device = get_device()
        print(f'DEVICE: {self.device}')
        # 内部参数
        self.ll = Path(model_path).parent.name.split('_')
        print(self.ll)
        self.set_file = str(Path(set).joinpath(self.ll[2], 'test'))
        self.model_path = model_path
        # 训练参数
        self.kind = list(path.name for path in Path(self.set_file).iterdir())  # if path.is_dir()
        self.norm_type = self.ll[1]
        self.set = file_dataset(self.set_file, self.norm_type, augment=False, stack=stack)
        self.val_set, _ = self.set.data_preprare(val_size=0)
        self.batch_size = 55
        self.val_loader = DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False)
        # 模型
        input = np.load(str(next(Path(self.set_file).rglob('*.npy')))).shape[1]
        self.model_name = self.ll[0]
        self.net = getattr(model, self.model_name)(in_channel=self.set.inputchannel,
                                                   out_channel=self.set.num_classes).to(self.device)  # 输入和输出通道数
        self.net.load_state_dict(torch.load(model_path))  #
        # 记录器
        self.val_acc = 0.0
        self.false = {'index': [], 'label': [], 'data': [], 'pred': []}
        self.num_false = 0
        self.y_p = []
        self.y_t = []

    def set_up(self, detail=True):
        self.net.eval()  # set the model to evaluation mode
        with torch.no_grad():
            for i, data in enumerate(self.val_loader):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.net(inputs)
                _, val_pred = torch.max(outputs, 1)  # get the index of the class with the highest probability

                self.y_p += val_pred.tolist()
                self.y_t += labels.tolist()
                right_pred = val_pred.cpu() == labels.cpu()
                self.val_acc += right_pred.sum().item()
                if detail:
                    false_pred = val_pred.cpu() != labels.cpu()
                    self.false['index'].append(torch.tensor(np.where(false_pred)[0] + i*self.batch_size))
                    self.false['label'].append(labels.cpu()[false_pred])
                    self.false['data'].append(inputs.cpu()[false_pred])
                    self.false['pred'].append(val_pred.cpu()[false_pred])
            self.val_acc = self.val_acc / len(self.val_set)
            if detail:
                for key in self.false.keys():
                    self.false[key] = torch.cat(self.false[key], dim=0).numpy()  # 被错分的数据的长度(false_num, 1, 4096)
                self.num_false = len(self.false['label'])
        return self.val_acc, self.false, self.y_p, self.y_t

    def confusion(self, y_t=None, y_p=None, is_save=True):
        if y_t == None:
            y_t = self.y_t
        if y_p == None:
            y_p = self.y_p
        conf_matrix = confusion_matrix(y_t, y_p)
        # plt.figure(1)
        heatmap = seaborn.heatmap(pd.DataFrame(conf_matrix, index=self.kind, columns=self.kind), annot=True, fmt='.5g',
                                  cmap='Blues')  #
        plt.title('the model ' + Path(self.model_path).stem + ' with accuracy: ' + str(self.val_acc))
        if is_save:
            plt.savefig(Path(self.model_path).parent / (str(Path(self.model_path).stem) + '.png'))
        plt.show()

    def detial_1d_5(self, index=None, label=None, pred=None, signal=None):  # 未定义索引将随机绘制5个
        index = (np.random.randint(self.num_false, size=5) if index == None else index)
        label = (self.false['label'][index] if label == None else label)
        pred = (self.false['pred'][index] if pred == None else pred)
        signal = (np.squeeze(self.false['data'][index], 1) if signal == None else signal)

        def time(i):
            t = np.linspace(0, len(signal[i]) * 0.004, len(signal[i]))  # 定义时间轴尺度
            plt.plot(t, signal[i], linewidth=0.5)
            plt.title('real class:' + self.kind[label[i]] + ', pred class:' + self.kind[pred[i]] + '\ntime domain')

        def freq(i):
            y = npfft.fft(signal[i])
            # print(len(y)==len(signal), len(y), len(signal))# True 4096 4096
            f = np.linspace(0, 0.5 / 0.004, len(signal[i]) // 2)  # 定义频率轴尺度，最高分析频率范围
            plt.plot(f, np.absolute(y)[0:len(y) // 2] * 2 / len(y), linewidth=0.5)  # 计算幅值谱，并切片出出现频率混叠之前的分量
            plt.title('frequency domain')

        num_rows = 5
        num_cols = 1
        plt.figure(figsize=(2 * 2 * 2 * num_cols, 2 * num_rows))
        for i in range(5):
            plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
            time(i)
            plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
            freq(i)
        plt.tight_layout()
        plt.show()

    def detial_1d(self, index=None, label=None, pred=None, signal=None):  # 未定义索引将随机绘制1个
        index = (int(np.random.randint(self.num_false, size=1)) if index == None else index)
        label = (self.false['label'][index] if label == None else label)
        pred = (self.false['pred'][index] if pred == None else pred)
        signal = (self.false['data'][index][0] if signal == None else signal)

        plt.subplot(2, 1, 1)
        t = np.linspace(0, len(signal) * 0.004, len(signal))  # 定义时间轴尺度
        plt.plot(t, signal, linewidth=0.5)
        plt.title('real class:' + self.kind[label] + ', pred class:' + self.kind[pred] + '\ntime domain')
        plt.subplot(2, 1, 2)
        y = npfft.fft(signal)
        # print(len(y)==len(signal), len(y), len(signal))# True 4096 4096
        f = np.linspace(0, 0.5 / 0.004, len(signal) // 2)  # 定义频率轴尺度，最高分析频率范围
        plt.plot(f, np.absolute(y)[0:len(y) // 2] * 2 / len(y), linewidth=0.5)  # 计算幅值谱，并切片出出现频率混叠之前的分量
        plt.title('frequency domain')
        plt.show()

    def detail_2d(self, index=None, label=None, pred=None, signal=None, cmap='rainbow'):
        index = (int(np.random.randint(self.num_false, size=1)) if index == None else index)
        label = (self.false['label'][index] if label == None else label)
        pred = (self.false['pred'][index] if pred == None else pred)
        signal = (self.false['data'][index][0] if signal == None else signal)

        plt.pcolormesh(signal, shading='auto', cmap=cmap)
        plt.title(
            'index[' + str(index) + '] real class:' + self.kind[label] + ', pred class:' + self.kind[pred] + '\nSTFT')
        plt.show()


if __name__ == '__main__':
    tt = tester(
        model_path=r'C:\Users\Yipin\OneDrive\program\blade_turbine_fd\checkpoint\ecnnfft_1-1_demo4fft_0923-130514\57-0.8052-best_model.pth')
    tt.set_up()  # 获取预测数据
    tt.confusion(is_save=True)  # 绘制混淆矩阵
    # tt.detial_1d_5()# 绘制5个时域图和频谱图
    # tt.detial_1d()# 绘制时域图和频谱图
