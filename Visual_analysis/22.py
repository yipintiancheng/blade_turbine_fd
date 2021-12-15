import torch, torchvision
from torch import nn
from torchvision import transforms, models, datasets
import shap
import json
import numpy as np
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def normalize(image):
    if image.max() > 1:
        image /= 255
    image = (image - mean) / std
    # in addition, roll the axis so that they suit pytorch
    return torch.tensor(image.swapaxes(-1, 1).swapaxes(2, 3)).float()
# load the model
model = models.vgg16(pretrained=True).eval()

X,y = shap.datasets.imagenet50()

X /= 255

to_explain = X[[39, 41]]

# load the ImageNet class names
url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
fname = shap.datasets.cache(url)
with open(fname) as f:
    class_names = json.load(f)
class_names0 = [s[1] for s in class_names.values()]

# python function to get model output; replace this function with your own model function.

def f(x):
    tmp = x.copy()
#     print(model(normalize(tmp)).shape)
#     print(model(normalize(tmp)).sum)
    return model(normalize(tmp))
print(fname)

with torch.no_grad():
    model = models.vgg16(pretrained=True).eval()

    # define a masker that is used to mask out partitions of the input image.
    masker = shap.maskers.Image("inpaint_telea", X[0].shape)
    # masker_blur = shap.maskers.Image("blur(128,128)", X[0].shape)

    # create an explainer with model and image masker
    explainer = shap.Explainer(f, masker, output_names=class_names0)#

    # here we explain two images using 500 evaluations of the underlying model to estimate the SHAP values
    shap_values = explainer(X[[39,41]], max_evals=100, batch_size=50, outputs=shap.Explanation.argsort.flip[:3])
    # output with shap values
    shap.image_plot(shap_values)