import numpy as np
import torch
import os
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from utils.dataset.file2dataset import file_dataset
import model#.ecnn import ECNN as ECNN


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'#


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def plott(train_accl, train_lossl, val_accl, val_lossl):
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(train_accl, label='Training Accuracy')
    plt.plot(val_accl, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    
    plt.subplot(2, 1, 2)
    plt.plot(train_lossl, label='Training Loss')
    plt.plot(val_lossl, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    
    plt.show()


# 设置参数
same_seeds(0)
device = get_device()
print(f'DEVICE: {device}')
set_file = 'E:\\dataset\\demo2\\train'
num_epoch = 60
batch_size = 32
learning_rate = 0.0001
model_path = './checkpoint/model.pth'

# 搭建模型
train_set, val_set = file_dataset(set_file, 'mean-std').data_preprare()#拆分数据集#默认val_size=0.2, random_state=40, test=False, mode='val'
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
# create model, define a loss function, and optimizer
model = getattr(model, 'ecnn')(len(os.listdir(set_file))).to(device)
criterion = nn.CrossEntropyLoss()
optimizer_StepLR = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
StepLR = torch.optim.lr_scheduler.StepLR(optimizer_StepLR, step_size=20, gamma=0.2)

# 训练循环
best_acc = 0.0#用来标记最好的那次训练结果
train_accl = []
train_lossl = []
val_accl = []
val_lossl = []
for epoch in range(num_epoch):
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0
    
    # training
    model.train()  # set the model to training mode
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer_StepLR.zero_grad()
        outputs = model(inputs)
        batch_loss = criterion(outputs, labels)#计算交叉熵前不用经过softmax
        _, train_pred = torch.max(outputs, 1)  # get the index of the class with the highest probability
        batch_loss.backward()
        optimizer_StepLR.step()
        StepLR.step()
        
        train_acc += (train_pred.cpu() == labels.cpu()).sum().item()#统计本个epoch里所有batch计算后的准确的个数
        train_loss += batch_loss.item()#统计本个epoch里所有batch计算后的损失总和
    
    # validation#由验证集决定保存哪个模型
    if len(val_set) > 0:#判断是否有验证集
        model.eval()  # set the model to evaluation mode
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                batch_loss = criterion(outputs, labels)
                _, val_pred = torch.max(outputs, 1)
                
                val_acc += (val_pred.cpu() == labels.cpu()).sum().item()
                val_loss += batch_loss.item()
            
            (ta,tl,va,vl) = (train_acc / len(train_set), train_loss / len(train_loader), val_acc / len(val_set), val_loss / len(val_loader))
            print('epoch:[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}  |  Val Acc: {:3.6f} loss: {:3.6f}'.format(epoch + 1, num_epoch, ta, tl, va, vl))
            # 验证集的准确率和平均损失（每个batch的平均）#一个epoch结束后的准确率和平均损失（每个batch的平均）
            train_accl.append(ta)
            train_lossl.append(tl)
            val_accl.append(va)
            val_lossl.append(vl)
            
            if val_acc > best_acc:#判断是否更新保存的模型（由准确率决定）
                best_acc = val_acc
                torch.save(model.state_dict(), model_path)
                print('---*** 本次epochs后的模型准确率{:.3f}为历史最好 ***---'.format(best_acc / len(val_set)))
    else:
        (ta, tl) = (train_acc / len(train_set), train_loss / len(train_loader))
        print('epoch:[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format(
            epoch + 1, num_epoch, ta, tl))#一个epoch结束后的准确率和平均损失（每个batch的平均）
        train_accl.append(ta)
        train_lossl.append(tl)
    
# 保存模型及可视化训练过程
if len(val_set) == 0:#当没有验证集时会保存最后一次训练的结果
    torch.save(model.state_dict(), model_path)
    print('saving model at last epoch')
plott(train_accl, train_lossl, val_accl, val_lossl)