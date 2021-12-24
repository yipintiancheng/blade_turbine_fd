#!/usr/bin/python
# -*- coding:utf-8 -*-

import argparse

import numpy as np

from utils.test_utils import tester

args = None


def parse_args():  # 获得命令行参数
    parser = argparse.ArgumentParser(description='Train')  # 参数对象

    # basic parameters
    parser.add_argument('--model_path', type=str,
                        default=r'E:\dataset\checkpoint\resnet182d_0-1_demo73stft_1223-151825\17-1.0000-best_model.pth',
                        help='the path of the model')  # 可直接更改这个默认参数，进行测试
    # other parameters
    parser.add_argument('--data_dir', type=str, default="E:\\dataset", help='the directory of the data')
    parser.add_argument('--input_channel', type=int, default=1, help='一般都是1不用改，这个还是手动改吧，如果training.log文件里的pretrained=True,那这里的stack就是3')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    tester = tester(set=args.data_dir, model_path=args.model_path, stack=args.input_channel)
    tester.set_up(detail=False)  # 获取预测数据# detail=False可以提高速度
    tester.confusion(is_save=True)  # 绘制混淆矩阵

    # 绘制特定条件被分错的图像
    # index_label = np.where(tester.false['label'] == 5)[0]
    # index_pred = np.where(tester.false['pred'] == 1)[0]
    # index = list(
    #     set(np.where(tester.false['label'] == 5)[0]) & set(np.where(tester.false['pred'] == 1)[0]))
    # for i in range(5):
    #     tester.detail_2d(index=index[i])

    # tester.detial_1d_5()# 绘制5个时域图和频谱图
    # tester.detial_1d()# 绘制时域图和频谱图
