import random
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from abc import ABC, abstractmethod

## 基础数据集类定义
class BaseDataset(data.Dataset, ABC):
    """此类是数据集的抽象基类（ABC）。
    要创建子类，您需要实现以下四个函数：
    -- < __init__ >：初始化类，首先调用BaseDataset__init__（self，opt）。
    -- < __len__ >：返回数据集的大小。
    -- < __getitem__ >：获取一个数据点。
    --＜modify_commandline_options＞：（可选）添加特定于数据集的选项并设置默认选项。
    """
    def __init__(self, opt):
        # 初始化类；保存类中的选项
        # 参数：
        # opt（Option类）——存储所有experiment flags；需要是BaseOptions的子类
        self.opt = opt
        self.root = opt.dataroot

    @staticmethod
    def modify_commandline_options(parser, is_train):
        # 添加新的特定于数据集的选项，并重写现有选项的默认值。
        # 参数：
        # parser—原始选项解析器
        # is_train（bool）——无论是训练阶段还是测试阶段。您可以使用此标志添加特定于培训或特定于测试的选项。
        # 返回：
        # 修改后的解析器。
        return parser

    @abstractmethod
    def __len__(self):
        """返回数据集中的图像总数."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        # 返回数据点及其元数据信息。
        # 参数：
        # index - -用于数据索引的随机整数
        # 返回：
        # 一个有他们名字的数据字典。它通常包含数据本身及其元数据信息。
        pass

## 返回数据增强的参数
def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.preprocess == 'resize_and_crop':
        new_h = new_w = opt.load_size
    elif opt.preprocess == 'scale_width_and_crop':
        new_w = opt.load_size
        new_h = opt.load_size * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

    flip = random.random() > 0.5

    return {'crop_pos': (x, y), 'flip': flip}

## 预处理操作
def get_transform(opt, params=None, grayscale=False, method=Image.BICUBIC, convert=True):
    """
    返回预处理操作的图像变换列表。

    参数：
        opt (Option类) - 存储所有实验标志的类；需要是BaseOptions的子类
        params (dict) - 数据增强的参数，包括裁剪位置和翻转
        grayscale (bool) - 是否将图像转为灰度图
        method (int) - 图像缩放的插值方法，默认为BICUBIC
        convert (bool) - 是否执行图像的标准化操作；

    返回：
        transforms.Compose(transform_list) - 一个包含预处理操作的图像变换组合
    """
    transform_list = []

    # 如果需要将图像转为灰度图，则添加灰度转换操作
    if grayscale:
        transform_list.append(transforms.Grayscale(1))

    # 根据预处理选项进行相应的图像变换
    if 'resize' in opt.preprocess:
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize, method))  # 使用指定的插值方法进行图像resize
    elif 'scale_width' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, opt.crop_size, method)))  # 根据指定的宽度进行缩放

    if 'crop' in opt.preprocess:
        if params is None:
            transform_list.append(transforms.RandomCrop(opt.crop_size))  # 随机裁剪图像
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))  # 根据给定参数裁剪图像

    if opt.preprocess == 'none':
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))  # 调整图像大小为2的幂次方

    if not opt.no_flip:  # 如果不禁用翻转
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())  # 随机水平翻转图像
        elif params['flip']:
            transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))  # 根据给定参数翻转图像

    if convert:  # 如果需要标准化
        transform_list += [transforms.ToTensor()]  # 转为Tensor
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]  # 灰度图像的标准化操作
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]  # RGB图像的标准化操作

    return transforms.Compose(transform_list)  # 返回一个包含所有预处理操作的组合


## 图像缩放
def __make_power_2(img, base, method=Image.BICUBIC):# 把高度和宽度变成2的指数次方
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)

## 基于宽度的图像缩放
def __scale_width(img, target_size, crop_size, method=Image.BICUBIC):
    ow, oh = img.size
    if ow == target_size and oh >= crop_size:
        return img
    w = target_size # 指定宽度
    h = int(max(target_size * oh / ow, crop_size))# 保证新的高度不小于cropsize
    return img.resize((w, h), method)

## 图像裁剪
def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img

## 图像翻转
def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True
