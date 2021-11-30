import numpy as np
import model
import torch
import os
from pathlib import Path
import seaborn
import matplotlib.pyplot as plt
import numpy.fft as npfft
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from utils.dataset.file2dataset import file_dataset


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


# # 获取模型（权重路径和模型类名）和数据集（路径）
# device = get_device()
# print(f'DEVICE: {device}')
# set_file = 'E:\dataset\demo2\\test'
# val_set, _ = file_dataset(set_file, 'mean-std').data_preprare(val_size=0)
# val_loader = DataLoader(val_set, batch_size=55, shuffle=False)
# net = getattr(model, 'ecnn')(1,7).to(device)
# net.load_state_dict(
#     torch.load(r'E:\blade_turbine_fd\checkpoint\ecnn_demo2_0917-163602\88-0.9740-best_model.pth')
#     )
# #print(net)
#
# # 计算出val_acc和p(预测矩阵)
# val_acc = 0.0
# false = {'label': [], 'data': [], 'pred': []}
# y_p = []
# y_t = []
# net.eval() # set the model to evaluation mode
# with torch.no_grad():
#     for i, data in enumerate(val_loader):
#         inputs, labels = data
#         inputs, labels = inputs.to(device), labels.to(device)
#         outputs = net(inputs)
#         _, val_pred = torch.max(outputs, 1) # get the index of the class with the highest probability
#
#         y_p += val_pred.tolist()
#         y_t += labels.tolist()
#         right_pred = val_pred.cpu() == labels.cpu()
#         false_pred = val_pred.cpu() != labels.cpu()
#         val_acc += right_pred.sum().item()
#         # false['label'].append(labels.cpu()[false_pred])
#         # false['data'].append(inputs.cpu()[false_pred])
#         # false['pred'].append(val_pred.cpu()[false_pred])
#     val_acc = val_acc / len(val_set)
#     # for key in false.keys():
#     #     false[key] = torch.cat(false[key], dim=0).numpy()#被错分的数据的长度(false_num, 1, 4096)
#
# # 绘制混淆矩阵
# conf_matrix = confusion_matrix(y_t, y_p)
# plt.figure(1)
# plt.title('accuracy : '+str(val_acc))
# heatmap = seaborn.heatmap(conf_matrix, annot=True,fmt='.5g', cmap='Blues')#
# #plt.savefig('heatmap.png')
# plt.show()

# 绘制细节（时域图，频域图，概率分布图）
index = 100
plt.figure(2)

plt.subplot(211)
signal = false['data'][index][0]# signal = np.cos(np.arange(0,0.004*4096,0.004))
t = np.linspace(0, len(signal)*0.004, len(signal))#定义时间轴尺度
plt.plot(t, signal, linewidth=0.5)

plt.subplot(212)
y = npfft.fft(signal)
#print(len(y)==len(signal), len(y), len(signal))# True 4096 4096
f = np.linspace(0, 0.5/0.004, len(signal)//2)# 定义频率轴尺度，最高分析频率范围
plt.plot(f, np.absolute(y)[0:len(y)//2]*2/len(y), linewidth=0.5)# 计算幅值谱，并切片出出现频率混叠之前的分量
plt.title('real class:'+str(false['label'][index])+'--pred class:'+str(false['pred'][index]))

plt.show()