from pathlib import Path
import model
import cv2
import torch
import numpy as np
from utils.dataset.file2dataset import file_dataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch, torchvision
from torch import nn
import torch.nn.functional as F
from torchvision import transforms, models, datasets
import shap
import json
import numpy as np


def initial(model_path):  # 初始话测试集和模型，现在先用3通道的输入,好像也不是
    f = Path(model_path)
    ll = f.parent.name.split('_')
    set_file = str(Path('E:\dataset').joinpath(ll[2], 'test'))
    norm_type = ll[1]
    model_name = ll[0]
    set = file_dataset(set_file, norm_type, augment=False)
    val_set, _ = set.data_preprare(val_size=0)
    net = getattr(model, model_name)(in_channel=set.inputchannel, out_channel=set.num_classes)
    net.load_state_dict(torch.load(model_path))
    return val_set, net


def sampler(val_set, loader=False):
    if loader:
        val_loader = DataLoader(val_set, batch_size=5, shuffle=False)
        sample, lab = next(iter(val_loader))
        print(lab)
    a = val_set[1300:1320]  # 66666-66666
    # a = val_set[77:82]#00000-06000
    # a = val_set[900:905]#44444-22222
    # a = val_set[850:855]#44444-44444
    # a = val_set[400:405]  # 22222-45552
    # a = val_set[405:410]  # 22222-24224
    sample = a[0].swapaxes(3, 1).swapaxes(2, 1)  # torch.tensor(a[0])
    # print(a.shape)
    print(a[1])
    return np.ascontiguousarray(a[0].swapaxes(3, 1).swapaxes(2, 1)), a[1]


def f(x):
    tmp = x.copy()
    # print(model(torch.tensor(tmp.swapaxes(1, 3).swapaxes(2, 3))).shape)
    # print(model(torch.tensor(tmp.swapaxes(1, 3).swapaxes(2, 3)).float()))
    return model(torch.tensor(tmp.swapaxes(1, 3).swapaxes(2, 3)).float())


def norm(x):
    return torch.tensor(x.swapaxes(1, 3).swapaxes(2, 3)).float()


def f1():
    with torch.no_grad():
        model = model.eval()

        # define a masker that is used to mask out partitions of the input image.
        masker = shap.maskers.Image("inpaint_telea", sample[0].shape)
        # masker_blur = shap.maskers.Image("blur(128,128)", X[0].shape)

        # create an explainer with model and image masker
        explainer = shap.Explainer(f, masker, output_names=class_names0)  #
        print(123)
        # here we explain two images using 500 evaluations of the underlying model to estimate the SHAP values
        shap_values = explainer(sample[[1, 4]], max_evals=100, batch_size=50,
                                outputs=shap.Explanation.argsort.flip[:3])
        # output with shap values/
        shap_values.output_names = np.asarray([['1', '1', '1'], ['1', '1', '1']])
        shap.image_plot(shap_values)


def f2():
    to_explain = sample[[1, 4]]
    e = shap.GradientExplainer((model, target_layers), norm(sample))  # , local_smoothing=0.5
    shap_values, indexes = e.shap_values(norm(to_explain), ranked_outputs=2, nsamples=200)

    # get the names for the classes
    index_names = np.vectorize(lambda x: class_names0[int(x)])(indexes)

    # plot the explanations
    shap_values = [np.swapaxes(np.swapaxes(s, 2, 3), 1, -1) for s in shap_values]

    shap.image_plot(shap_values, to_explain, index_names)


def f3():
    background = sample  # print(background.shape)
    to_explain = sample[[1, 4]]
    e = shap.DeepExplainer((model, target_layers), norm(background))
    shap_values, indexes = e.shap_values(norm(to_explain), ranked_outputs=2)
    shap_values = [np.swapaxes(np.swapaxes(s, 2, 3), 1, -1) for s in shap_values]
    print(shap_values[0].shape)
    index_names = np.vectorize(lambda x: class_names0[int(x)])(indexes)
    shap.image_plot(shap_values, to_explain, index_names)


model_path = r'E:\dataset\checkpoint\resnet182d_0-1_demo73stft_1014-160602\42-1.0000-best_model.pth'
val_set, model = initial(model_path)
target_layers = model.layer4[-1]
class_names0 = ['apex_drop', 'basic', 'edge_cut', 'girder_defects', 'ice_45.4m', 'ice_67.4m', 'ice_apex']
sample, label = sampler(val_set)  # 在数据集总采样
model.eval()
# outputs = model(input_tensor)
# grad, val_pred = torch.max(outputs, 1)
# print(val_pred.numpy())
print(type(model.layer4[-1]))
print(target_layers)

f3()
