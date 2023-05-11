import os
import random

import numpy as np
from PIL import Image, ImageFilter

import torch
import torchvision.transforms as transforms


def addGaussianNoise(x, args):
    return x + torch.randn(x.size()).cuda(args.gpu, non_blocking=True) * args.noise_std


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.1):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={0.1})'.format(self.mean, self.std)


class Transform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.8, contrast=0.8,
                                        saturation=0.8, hue=0.2)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            PILRandomGaussianBlur(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.228, 0.224, 0.225])
        ])

    def __call__(self, x):
        x1 = self.transform(x)
        return x1


class TransformWithNoise:
    def __init__(self, noise_std):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.8, contrast=0.8,
                                        saturation=0.8, hue=0.2)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            PILRandomGaussianBlur(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.228, 0.224, 0.225]),
            AddGaussianNoise(std=noise_std),
        ])

    def __call__(self, x):
        x1 = self.transform(x)
        return x1
    

class PILRandomGaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image. Take the radius and probability of
    application as the parameter.
    This transform was used in SimCLR - https://arxiv.org/abs/2002.05709
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = np.random.rand() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )
    
class Transform_noaug:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.228, 0.224, 0.225])
        ])

    def __call__(self, x):
        x1 = self.transform(x)
        return x1


class Transform_noaugWithNoise:
    def __init__(self,noise_std):
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.228, 0.224, 0.225]),
            AddGaussianNoise(std=noise_std),
        ])
    def __call__(self, x):
        x1 = self.transform(x)
        return x1


# TODO: not a good practice;
def zcaTransformation(x,args):
    transform = transforms.Compose([
        transforms.LinearTransformation(
            torch.tensor([[1.6098618, -1.0153007, -0.02540025],
                          [-1.0153007, 2.5014492, -1.04380666],
                          [-0.02540025, -1.04380666, 1.52629735]]).cuda(args.gpu, non_blocking=True),
            torch.tensor([-0.0879, -0.0229, 0.0722]).cuda(args.gpu, non_blocking=True)
        )
    ])
    batch_size = x.size()[0]
    zca_x = transform(x.transpose(1, 0).reshape(1, 1, 3, -1).T)
    zca_x = zca_x.T.view(3, batch_size, 224, 224).transpose(1, 0)
    return zca_x


#
# class ZCATransformation:
#     def __init__(self):
#         self.transform = transforms.Compose([
#             transforms.LinearTransformation(
#                 torch.tensor([[ 1.6098618 , -1.0153007 , -0.02540025],
#                    [-1.0153007 ,  2.5014492 , -1.04380666],
#                    [-0.02540025, -1.04380666,  1.52629735]]),
#                 torch.tensor([-0.0879, -0.0229,  0.0722])
#             )
#         ])
#     def __call__(self, x):
#         """
#         :param x: B*C*H*W
#         :return: B*C*H*W; pixel-level ZCA
#         """
#         batch_size = x.size()[0]
#         zca_x = self.transform(x.transpose(1,0).reshape(1,1,3,-1).T)
#         # TODO: not a good practice;
#         zca_x = zca_x.T.view(3,batch_size,224,224).transpose(1,0)
#         return zca_x




#
# """https://github.com/semi-supervised-paper/semi-supervised-paper-implementation/blob/e39b61ccab/semi_supervised/core/utils/data_util.py#L150"""
# class ZCATransformation(object):
#     def __init__(self, transformation_matrix, transformation_mean):
#         if transformation_matrix.size(0) != transformation_matrix.size(1):
#             raise ValueError("transformation_matrix should be square. Got " +
#                              "[{} x {}] rectangular matrix.".format(*transformation_matrix.size()))
#         self.transformation_matrix = transformation_matrix
#         self.transformation_mean = transformation_mean
#
#     def __call__(self, tensor):
#         """
#         Args:
#             tensor (Tensor): Tensor image of size (N, C, H, W) to be whitened.
#         Returns:
#             Tensor: Transformed image.
#         """
#         if tensor.size(1) * tensor.size(2) * tensor.size(3) != self.transformation_matrix.size(0):
#             raise ValueError("tensor and transformation matrix have incompatible shape." +
#                              "[{} x {} x {}] != ".format(*tensor[0].size()) +
#                              "{}".format(self.transformation_matrix.size(0)))
#         batch = tensor.size(0)
#
#         flat_tensor = tensor.view(batch, -1)
#         transformed_tensor = torch.mm(flat_tensor - self.transformation_mean, self.transformation_matrix)
#
#         tensor = transformed_tensor.view(tensor.size())
#         return tensor
#
#     def __repr__(self):
#         format_string = self.__class__.__name__ + '('
#         format_string += (str(self.transformation_matrix.numpy().tolist()) + ')')
#         return format_string
#
# """
# https://stackoverflow.com/questions/31528800/how-to-implement-zca-whitening-python
# """
# def zca_whitening_matrix(X):
#     sigma = np.cov(X, rowvar=False)
#     U,S,V = np.linalg.svd(sigma)
#     epsilon = 1e-5
#     ZCAMatrix = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + epsilon)), U.T))
#     return ZCAMatrix
#
#
# def zca_within_batch(tensor):
#     """
#     Args:
#         tensor (Tensor): Tensor image of size (N, C, H, W) to be whitened.
#     Returns:
#         Tensor: Transformed image.
#     """
#     batch = tensor.size(0)
#     flat_tensor = tensor.view(batch, -1)
#     transformation_matrix = zca_whitening_matrix(flat_tensor.numpy())
#     transformation_matrix = torch.from_numpy(transformation_matrix)
#     transformed_tensor = torch.mm(flat_tensor, transformation_matrix)
#     tensor = transformed_tensor.view(tensor.size())
#     return tensor
#
# """
# X = np.array([[0, 2, 2], [1, 1, 0], [2, 0, 1], [1, 3, 5], [10, 10, 10] ]) # Input: X [5 x 3] matrix
# ZCAMatrix = zca_whitening_matrix(X) # get ZCAMatrix
# ZCAMatrix # [5 x 5] matrix
# xZCAMatrix = np.dot(ZCAMatrix, X) # project X onto the ZCAMatrix
# xZCAMatrix # [5 x 3] matrix
# """
#
# def add_noise_and_clip_data(data):
#     noise = np.random.normal(loc=0.0, scale=0.1, size=data.shape)
#     data = data + noise
#     data = np.clip(data, 0., 1.)
#     return data
#
