"""该模块包含了一些简单的辅助函数"""
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os


def tensor2im(input_image, imtype=np.uint8):
    """"将一个张量数组转换为一个numpy图像数组。

    参数:
        input_image (tensor) -- 输入图像张量数组
        imtype (type)        -- 所需的转换后numpy数组的类型
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # 从变量中获取数据
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # 转换为numpy数组
        if image_numpy.shape[0] == 1:  # 灰度转RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # 后处理: 转置和缩放
    else:  # 如果是numpy数组，不做任何操作
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """计算并打印平均绝对梯度的均值

    参数:
        net (torch network) -- Torch网络
        name (str) -- 网络的名称
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """将一个numpy图像保存到磁盘

    参数:
        image_numpy (numpy array) -- 输入numpy数组
        image_path (str)          -- 图像的路径
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """打印numpy数组的平均值、最小值、最大值、中位数、标准差和大小

    参数:
        val (bool) -- 是否打印numpy数组的值
        shp (bool) -- 是否打印numpy数组的形状
    """
    x = x.astype(np.float64)
    if shp:
        print('形状,', x.shape)
    if val:
        x = x.flatten()
        print('均值 = %3.3f, 最小值 = %3.3f, 最大值 = %3.3f, 中位数 = %3.3f, 标准差=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """如果不存在，创建空目录

    参数:
        paths (str list) -- 目录路径的列表
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """如果不存在，创建单个空目录

    参数:
        path (str) -- 单个目录路径
    """
    if not os.path.exists(path):
        os.makedirs(path)
