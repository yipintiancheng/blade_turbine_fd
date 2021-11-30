from pathlib import Path
import model
import cv2
import torch
import numpy as np
from utils.dataset.file2dataset import file_dataset
import matplotlib.pyplot as plt
import pytorch_grad_cam
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch.utils.data import DataLoader
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image


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


def figure_source(val_set):
    testensor = [val_set[0], val_set[120], val_set[340], val_set[560], val_set[780], val_set[1000]]
    print([i[1] for i in testensor])
    plt.subplot(2, 3, 1)
    plt.contourf(testensor[0][0][2])
    plt.title(testensor[0][1])
    plt.subplot(2, 3, 2)
    plt.contourf(testensor[1][0][2])
    plt.title(testensor[1][1])
    plt.subplot(2, 3, 3)
    plt.contourf(testensor[0][0][2] - testensor[1][0][2])
    plt.title('0-1')
    # plt.title(testensor[2][1])
    plt.subplot(2, 3, 4)
    plt.contourf(testensor[3][0][2])
    plt.title(testensor[3][1])
    plt.subplot(2, 3, 5)
    plt.contourf(testensor[4][0][2])
    plt.title(testensor[4][1])
    plt.subplot(2, 3, 6)
    plt.contourf(testensor[3][0][2] - testensor[4][0][2])
    # plt.title(testensor[5][1])
    plt.title('3-4')


def sampler(val_set, loader=False):
    if loader:
        val_loader = DataLoader(val_set, batch_size=5, shuffle=False)
        sample, lab = next(iter(val_loader))
        print(lab)
    a = val_set[1300:1305]#66666-66666
    # a = val_set[77:82]#00000-06000
    # a = val_set[900:905]#44444-22222
    # a = val_set[850:855]#44444-44444
    # a = val_set[400:405]  # 22222-45552
    # a = val_set[405:410]  # 22222-24224
    sample = torch.tensor(a[0])
    print(a[1])
    return sample, a[1]


def show(i, input_tensor, grayscale_cam, label, pred, classif, if_save=False, if_color=False):  # 绘制图
    grayscale_cam = grayscale_cam[i, :]
    rgb_img = input_tensor[i][0:3]
    # print(rgb_img.shape)
    # for i in range(rgb_img.shape[0]):
    #     rgb_img[i] = (rgb_img[i]-rgb_img[i].min())/(rgb_img[i].max()-rgb_img[i].min())
    rgb_img = rgb_img.permute([1, 2, 0]).numpy()  # 转换通道格式
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)  # 这里只能融合rgb图片，但要是分开显示的话就不需要一定是3通道模型了

    plt.subplot(2, 3, 1)
    plt.imshow(rgb_img)
    plt.title('lab:%d_pred:%d' % (label[i], pred[i]))
    plt.subplot(2, 3, 2)
    if if_color:
        grayscale_cam = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
        grayscale_cam = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        grayscale_cam = np.float32(heatmap) / 255
        print(grayscale_cam.shape)
    plt.imshow(grayscale_cam)
    plt.title("CAM:%d" % classif[i])
    plt.subplot(2, 3, 3)
    plt.imshow(visualization)
    plt.title("both source and CAM")
    plt.subplot(2, 3, 4)
    plt.gca().invert_yaxis()
    plt.contourf(input_tensor[i][0])
    plt.title("channel 0")
    plt.subplot(2, 3, 5)
    plt.gca().invert_yaxis()
    plt.contourf(input_tensor[i][1])
    plt.title('channel 1')
    plt.subplot(2, 3, 6)
    plt.gca().invert_yaxis()
    plt.contourf(input_tensor[i][2])
    plt.title('channel 2')
    if if_save:
        save_dir = Path(model_path).parent.joinpath(str(Path(model_path).stem) + '_cam')
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_dir) + '\\++num ' + str(i) + '-class ' + str(target_category) + '.jpg')
    plt.show()


def confuse_cam(sample, model, target_layers, cam_type='XGradCAM'):
    result = []  #
    input_tensor = torch.tensor([i[0] for i in sample])
    model.eval()
    outputs = model(input_tensor)
    print(outputs)
    grad, val_pred = torch.max(outputs, 1)
    print(val_pred.numpy())
    label = [int(i[1]) for i in sample]
    print(np.array(label))
    for la in [0, 1, 2, 3, 4, 5, 6]:
        cam = getattr(pytorch_grad_cam, cam_type)(model=model, target_layer=target_layers)
        grayscale_cam = cam(input_tensor=input_tensor, target_category=la, eigen_smooth=False)
        result.append(grayscale_cam)
    result = np.array(result)
    result = np.transpose(result, (1, 0, 2, 3))
    return result, val_pred.numpy(), np.array(label)


def plot_conf(result, pred, label):
    for i, p in enumerate(result):
        for j, pp in enumerate(p):
            plt.subplot(7, 7, 7 * i + j + 1)
            plt.axis('off')
            if j == label[i]:
                plt.gca().add_patch(plt.Rectangle((0, 0), 126, 126, fill=False, edgecolor='red', linewidth=2))
            if j == pred[i]:
                plt.gca().add_patch(plt.Rectangle((4, 4), 120, 120, fill=False, edgecolor='blue', linewidth=1))
            plt.imshow(pp)
            plt.xticks([]), plt.yticks([])
    plt.show()


model_path = r'E:\dataset\checkpoint\resnet182d_0-1_demo84stft_1101-193232\34-1.0000-best_model.pth'
val_set, model = initial(model_path)
target_layers = model.layer4[-1]  # 用来绘制特征图的层
# figure_source(val_set)

# 热力覆盖图
sample, label = sampler(val_set)  # 在数据集总采样
target_categorys = [6, 6, 6, 6, 6]
input_tensor = sample  # 第一通道与第二通道在在截断时与第三通道差距太大，导致归一化后第三通道的值离谱的大，已修复
model.eval()
outputs = model(input_tensor)
grad, val_pred = torch.max(outputs, 1)
print(val_pred.numpy())
# with GradCAM(model=model, target_layers=target_layers, use_cuda=args.use_cuda) as cam:
cam = GradCAM(model=model, target_layer=target_layers)
grayscale_cam = cam(input_tensor=input_tensor, target_category=target_categorys, eigen_smooth=False)
show(0, input_tensor, grayscale_cam, label, val_pred.numpy(), target_categorys)
show(1, input_tensor, grayscale_cam, label, val_pred.numpy(), target_categorys)
show(2, input_tensor, grayscale_cam, label, val_pred.numpy(), target_categorys)
show(3, input_tensor, grayscale_cam, label, val_pred.numpy(), target_categorys)
show(4, input_tensor, grayscale_cam, label, val_pred.numpy(), target_categorys)

# # 混淆热力图
# # 0123456-0123456
# sample0 = [val_set[20], val_set[120], val_set[340], val_set[560], val_set[780], val_set[1000], val_set[1220]]
# # 0123456-5123456
# sample1 = [val_set[105], val_set[120], val_set[340], val_set[560], val_set[780], val_set[1000], val_set[1220]]
# # 0000000-0000000
# sample2 = [val_set[5], val_set[10], val_set[20], val_set[40], val_set[80], val_set[90], val_set[100]]
# # 2222222-6266666
# sample3 = [val_set[500], val_set[501], val_set[502], val_set[503], val_set[504], val_set[505], val_set[506]]
# result, pred, label = confuse_cam(sample2, model, target_layers, cam_type='GradCAM')
# plot_conf(result, pred, label)

# # GuidedBackprop
# gb_model = GuidedBackpropReLUModel(model=model, use_cuda=True)
# gb = gb_model(input_tensor, target_category=4)
# grayscale_cam = grayscale_cam[0]
# cam_gb = deprocess_image(cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam]) * gb)
# gb = deprocess_image(gb)
# _, ax = plt.subplots(1, 2)
# ax[0].imshow(cam_gb)
# ax[0].set_title('target_category=4')
# ax[1].imshow(gb)
# plt.show()
