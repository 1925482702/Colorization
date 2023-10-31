"""模型类模板

该模块提供了一个模型类模板，用户可以使用它来实现自定义模型。
您可以指定'--model template'来使用此模型。
类名应与文件名和模型选项保持一致。
文件名应为<model>_dataset.py
类名应为<Model>Dataset.py
它实现了一个基于回归损失的简单图像到图像转换基线。
给定输入-输出对（data_A，data_B），它学习一个网络netG，可以最小化以下L1损失：
    min_<netG> ||netG(data_A) - data_B||_1
您需要实现以下函数：
    <modify_commandline_options>: 添加特定于模型的选项并重写现有选项的默认值。
    <__init__>: 初始化此模型类。
    <set_input>: 解包输入数据并执行必要的数据预处理。
    <forward>: 运行前向传递。将被<optimize_parameters>和<test>调用。
    <optimize_parameters>: 更新网络权重；将在每个训练迭代中调用。
"""

import torch
from .base_model import BaseModel
from . import networks

class TemplateModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """添加新的特定于模型的选项并重写现有选项的默认值。

        参数:
            parser -- 选项解析器
            is_train -- 是否为训练阶段。您可以使用此标志添加特定于训练或测试的选项。

        返回:
            修改后的选项解析器。
        """
        parser.set_defaults(dataset_mode='aligned')  # 您可以重写此模型的默认值。例如，此模型通常使用对齐的数据集作为其数据集。
        if is_train:
            parser.add_argument('--lambda_regression', type=float, default=1.0, help='回归损失的权重')  # 您可以为此模型定义新的参数。

        return parser

    def __init__(self, opt):
        """初始化此模型类。

        参数:
            opt -- 训练/测试选项

        这里可以做一些事情：
        -（必需）调用BaseModel的初始化函数
        - 定义损失函数、可视化图像、模型名称和优化器
        """
        BaseModel.__init__(self, opt)  # 调用BaseModel的初始化方法
        # 指定要打印的训练损失。程序将调用base_model.get_current_losses将损失绘制到控制台并保存到磁盘。
        self.loss_names = ['loss_G']
        # 指定要保存和显示的图像。程序将调用base_model.get_current_visuals来保存和显示这些图像。
        self.visual_names = ['data_A', 'data_B', 'output']
        # 指定要保存到磁盘的模型。程序将调用base_model.save_networks和base_model.load_networks来保存和加载网络。
        # 您可以使用opt.isTrain来指定训练和测试的不同行为。例如，一些网络在测试时不会被使用，因此您不需要加载它们。
        self.model_names = ['G']
        # 定义网络；您可以使用opt.isTrain来指定训练和测试的不同行为。
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, gpu_ids=self.gpu_ids)
        if self.isTrain:  # 仅在训练时定义
            # 定义您的损失函数。您可以使用torch.nn提供的损失，如torch.nn.L1Loss。
            # 我们还提供了一个GANLoss类“networks.GANLoss”。self.criterionGAN = networks.GANLoss().to(self.device)
            self.criterionLoss = torch.nn.L1Loss()
            # 定义和初始化优化器。您可以为每个网络定义一个优化器。
            # 如果同时更新两个网络，您可以使用itertools.chain将它们分组。请参阅cycle_gan_model.py作为示例。
            self.optimizer = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = [self.optimizer]

        # 我们的程序将自动调用<model.setup>来定义调度程序，加载网络并打印网络

    def set_input(self, input):
        """解包输入数据从数据加载器并执行必要的预处理。

        参数:
            input: 包含数据本身及其元数据信息的字典。
        """
        AtoB = self.opt.direction == 'AtoB'  # 使用<direction>来交换data_A和data_B
        self.data_A = input['A' if AtoB else 'B'].to(self.device)  # 获取图像数据A
        self.data_B = input['B' if AtoB else 'A'].to(self.device)  # 获取图像数据B
        self.image_paths = input['A_paths' if AtoB else 'B_paths']  # 获取图像路径

    def forward(self):
        """运行前向传递。将被<optimize_parameters>和<test>调用。"""
        self.output = self.netG(self.data_A)  # 给定输入数据data_A生成输出图像

    def backward(self):
        """计算损失、梯度并更新网络权重；在每个训练迭代中调用"""
        # 计算中间结果（如果需要）；这里self.output已在<forward>函数中计算
        # 根据输入和中间结果计算损失
        self.loss_G = self.criterionLoss(self.output, self.data_B) * self.opt.lambda_regression
        self.loss_G.backward()       # 计算网络G相对于损失_G的梯度

    def optimize_parameters(self):
        """更新网络权重；将在每个训练迭代中调用。"""
        self.forward()               # 首先调用forward计算中间结果
        self.optimizer.zero_grad()   # 清除网络G的现有梯度
        self.backward()              # 计算网络G的梯度
        self.optimizer.step()        # 更新网络G的梯度
