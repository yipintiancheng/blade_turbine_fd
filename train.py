#!/usr/bin/python
# -*- coding:utf-8 -*-

import argparse
import os
from datetime import datetime
from utils.logger import setlogger
import logging
from pathlib import Path
from utils.train_utils import train_utils
args = None


def parse_args():#获得命令行参数
    parser = argparse.ArgumentParser(description='Train')#参数对象
    # basic parameters
    parser.add_argument('--data_dir', type=str, default="E:/dataset", help='the directory of the data')
    parser.add_argument("--pretrained", type=bool, default=False, help='whether to load the pretrained model') # pretrain并不好用
    parser.add_argument('--cuda_device', type=str, default='0', help='assign device')
    parser.add_argument('--checkpoint_dir', type=str, default='E:/dataset/checkpoint', help='the directory to save the model')
    parser.add_argument('--num_workers', type=int, default=0, help='the number of training process')

    # optimization information
    parser.add_argument('--model_name', type=str,
                        default='resnet182d', help='the name of the model')  # 更改模型就用ecnnfft
    parser.add_argument('--data_name', type=str,
                        default='demo73stft', help='the name of the datafile')  # demo2这个数据集未被标准化,demo3好像是均值化过demo5cwt
    parser.add_argument('--normlizetype', type=str, choices=['0-1', '1-1', 'mean-std', 'None'],
                        default='0-1', help='data normalization methods')
    parser.add_argument('--augment', type=bool, default=True, help='whether to augment the train data')
    parser.add_argument('--batch_size', type=int,
                        default=32, help='batchsize of the training process')
    parser.add_argument('--opt', type=str, choices=['sgd', 'adam'],
                        default='adam', help='the optimizer')
    parser.add_argument('--lr', type=float,
                        default=0.0001, help='the initial learning rate')
    parser.add_argument('--momentum', type=float,
                        default=0.9, help='the momentum for sgd')
    parser.add_argument('--weight_decay', type=float,
                        default=3e-5, help='the weight decay')
    parser.add_argument('--lr_scheduler', type=str, choices=['step', 'exp', 'stepLR', 'fix'],
                        default='fix', help='the learning rate schedule')
    parser.add_argument('--gamma', type=float,
                        default=0.1, help='learning rate scheduler parameter for step and exp')
    parser.add_argument('--steps', type=str,
                        default='20', help='the learning rate decay for step and stepLR')
    
    # save, load and display information
    parser.add_argument('--max_epoch', type=int, default=20, help='max number of epoch')
    parser.add_argument('--print_step', type=int, default=10, help='the interval of log training information')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device.strip()
    # 创建文件夹
    sub_dir = args.model_name+'_'+args.normlizetype+'_'+args.data_name + '_' + datetime.strftime(datetime.now(), '%m%d-%H%M%S')
    save_dir = os.path.join(args.checkpoint_dir, sub_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)# 新建一个文件夹用来保存.log和.pth
    # 保存训练参数
    setlogger(os.path.join(save_dir, 'training.log'))
    for k, v in args.__dict__.items():
        logging.info("{}: {}".format(k, v))
    # 将命令行参数以及数据集路径传入,开始训练
    trainer = train_utils(args, save_dir)
    trainer.setup()
    trainer.train()
