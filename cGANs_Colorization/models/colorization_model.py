from .pix2pix_model import Pix2PixModel
import torch
from skimage import color  # 用于lab2rgb
import numpy as np


class ColorizationModel(Pix2PixModel):# 继承pix2pix
    """这是Pix2PixModel的子类，用于图像上色（黑白图像 -> 彩色图像）。

    模型训练需要 '-dataset_model colorization' 数据集。
    它训练一个 pix2pix 模型，将 L 通道映射到 Lab 色彩空间中的 ab 通道。
    默认情况下，上色数据集会自动设置 '--input_nc 1' 和 '--output_nc 2'。
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True): # 修改一些参数
        """添加新的特定于数据集的选项，并重写现有选项的默认值。

        参数：
            parser          -- 原始选项解析器
            is_train (bool) -- 是否处于训练阶段。您可以使用此标志添加特定于训练或测试的选项。

        返回：
            修改后的选项解析器。

        默认情况下，我们使用 'colorization' 数据集进行此模型。
        请参阅原始的 pix2pix 论文（https://arxiv.org/pdf/1611.07004.pdf）和上色结果（论文中的图表 9）
        """
        Pix2PixModel.modify_commandline_options(parser, is_train) # 调用pix2pix一些函数
        parser.set_defaults(dataset_mode='colorization') # 设置模式为colorization
        return parser

    def __init__(self, opt):
        """初始化类。

        参数：
            opt (Option类)-- 存储所有实验标志；需要是 BaseOptions 的子类

        对于可视化，我们将 'visual_names' 设置为 'real_A'（输入真实图像），
        'real_B_rgb'（真实的 RGB 图像）和 'fake_B_rgb'（预测的 RGB 图像）。
        我们将 Lab 图像 'real_B'（从 Pix2pixModel 继承）转换为 RGB 图像 'real_B_rgb'。
        我们将 Lab 图像 'fake_B'（从 Pix2pixModel 继承）转换为 RGB 图像 'fake_B_rgb'。
        """
        # 重用 pix2pix 模型
        Pix2PixModel.__init__(self, opt)
        ## 可视化图片
        self.visual_names = ['real_A', 'real_B_rgb', 'fake_B_rgb'] # 输入图、标签、生成的图片

    def lab2rgb(self, L, AB): # 将数据转换回 RGB
        """将 Lab 张量图像转换为 RGB numpy 输出
        参数：
            L  (1 通道张量数组): L 通道图像（范围：[-1, 1]，torch 张量数组）
            AB (2 通道张量数组): ab 通道图像（范围：[-1, 1]，torch 张量数组）

        返回：
            rgb (RGB numpy 图像): rgb 输出图像（范围：[0, 255]，numpy 数组）
        """
        AB2 = AB * 110.0    # ab 通道 * 110，反归一
        L2 = (L + 1.0) * 50.0 # 反标准化
        Lab = torch.cat([L2, AB2], dim=1) # 拼成 lab 格式
        Lab = Lab[0].data.cpu().float().numpy()
        Lab = np.transpose(Lab.astype(np.float64), (1, 2, 0))
        rgb = color.lab2rgb(Lab) * 255   # 转换回 RGB
        return rgb

    def compute_visuals(self): # 计算可视化
        """计算用于 visdom 和 HTML 可视化的附加输出图像"""
        self.real_B_rgb = self.lab2rgb(self.real_A, self.real_B)# 真实图像
        self.fake_B_rgb = self.lab2rgb(self.real_A, self.fake_B)# 预测图像
