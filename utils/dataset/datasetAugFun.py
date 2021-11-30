import numpy as np
import random
from scipy.signal import resample
from PIL import Image
import scipy
import torch


# import cv2

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, seq):
        for t in self.transforms:
            seq = t(seq)
        return seq


class Scale2d(object):
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, seq):
        scale_factor = np.random.normal(loc=1, scale=self.sigma, size=(seq.shape[0], 1))
        return seq * scale_factor


class RandomScale2d(object):
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, seq):
        if np.random.randint(2):
            return seq
        else:
            scale_factor = np.random.normal(loc=1, scale=self.sigma, size=(seq.shape[0], 1, 1))
            return seq * scale_factor


class RandomCrop2d(object):
    def __init__(self, crop_len=20):
        self.crop_len = crop_len

    def __call__(self, seq):
        if np.random.randint(2):
            return seq
        else:
            # print(seq.shape)
            max_height = seq.shape[1] - self.crop_len
            max_length = seq.shape[2] - self.crop_len
            random_height = np.random.randint(max_height)
            random_length = np.random.randint(max_length)
            seq[random_length:random_length + self.crop_len, random_height:random_height + self.crop_len] = 0
            return seq


class Reshape(object):
    def __call__(self, seq):
        # print(seq.shape)
        # print(seq.T.shape)
        return seq.T


class nothing(object):
    def __call__(self, seq):
        return seq


class Retype(object):
    def __call__(self, seq):
        return seq.astype(np.float32)


class AddGaussian(object):
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, seq):
        return seq + np.random.normal(loc=0, scale=self.sigma, size=seq.shape)


class RandomAddGaussian(object):
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, seq):
        if np.random.randint(2):
            return seq
        else:
            return seq + np.random.normal(loc=0, scale=self.sigma, size=seq.shape)


class Scale(object):
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, seq):
        scale_factor = np.random.normal(loc=1, scale=self.sigma, size=(seq.shape[0], 1))
        scale_matrix = np.matmul(scale_factor, np.ones((1, seq.shape[1])))
        return seq * scale_matrix


class RandomScale(object):
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, seq):
        if np.random.randint(2):
            return seq
        else:
            scale_factor = np.random.normal(loc=1, scale=self.sigma, size=(seq.shape[0], 1))
            scale_matrix = np.matmul(scale_factor, np.ones((1, seq.shape[1])))
            return seq * scale_matrix


class RandomStretch(object):
    def __init__(self, sigma=0.3):
        self.sigma = sigma

    def __call__(self, seq):
        if np.random.randint(2):
            return seq
        else:
            seq_aug = np.zeros(seq.shape)
            len = seq.shape[1]
            length = int(len * (1 + (random.random() - 0.5) * self.sigma))
            for i in range(seq.shape[0]):
                y = resample(seq[i, :], length)
                if length < len:
                    if random.random() < 0.5:
                        seq_aug[i, :length] = y
                    else:
                        seq_aug[i, len - length:] = y
                else:
                    if random.random() < 0.5:
                        seq_aug[i, :] = y[:len]
                    else:
                        seq_aug[i, :] = y[length - len:]
            return seq_aug


class RandomCrop(object):
    def __init__(self, crop_len=20):
        self.crop_len = crop_len

    def __call__(self, seq):
        if np.random.randint(2):
            return seq
        else:
            max_index = seq.shape[1] - self.crop_len
            random_index = np.random.randint(max_index)
            seq[:, random_index:random_index + self.crop_len] = 0
            return seq


class Normalize(object):
    def __init__(self, type=None):  # "0-1","1-1","mean-std"
        self.type = type

    def __call__(self, seq):
        if self.type == 'None':
            seq = seq
        elif self.type == "0-1":
            if len(seq.shape) > 2:
                seq = np.array([(i - i.min()) / (i.max() - i.min()) for i in seq])
            else:
                seq = (seq - seq.min()) / (seq.max() - seq.min())
        elif self.type == "1-1":
            if len(seq.shape) > 2:
                seq = np.array([2 * (i - i.min()) / (i.max() - i.min()) - 1 for i in seq])
            else:
                seq = 2 * (seq - seq.min()) / (seq.max() - seq.min()) - 1
        elif self.type == "mean-std":
            if len(seq.shape) > 2:
                seq = np.array([(i - i.mean()) / i.std() for i in seq])
            else:
                seq = (seq - seq.mean()) / seq.std()
        else:
            raise NameError('This normalization is not included!')
        return seq

    # def __call__(self, seq):
    #     if self.type == 'None':
    #         seq = seq
    #     elif self.type == "0-1":
    #         seq = (seq - seq.min()) / (seq.max() - seq.min())
    #     elif self.type == "1-1":
    #         seq = 2 * (seq - seq.min()) / (seq.max() - seq.min()) - 1
    #     elif self.type == "mean-std":
    #         seq = (seq - seq.mean()) / seq.std()
    #     else:
    #         raise NameError('This normalization is not included!')
    #     return seq
