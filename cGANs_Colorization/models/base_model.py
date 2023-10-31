import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from . import networks

class BaseModel(ABC):
    """这个类是一个抽象基类（ABC）用于模型。
    ## 需要额外实现的5个函数
        -- <__init__>:                      初始化类；首先调用BaseModel.__init__(self, opt)。
        -- <set_input>:                     从数据集解包数据并应用预处理。
        -- <forward>:                       生成中间结果。
        -- <optimize_parameters>:           计算损失，梯度并更新网络权重。
        -- <modify_commandline_options>:    （可选）添加模型特定选项并设置默认选项。
    """

    def __init__(self, opt):
        """初始化BaseModel类。

           参数：
               opt (Option类)-- 存储所有实验标志；需要是BaseOptions的子类。

           在创建您的自定义类时，您需要实现自己的初始化。
           在这个函数中，您应该首先调用<BaseModel.__init__(self, opt)>
           然后，您需要定义四个列表：
               -- self.loss_names（str列表）：指定要绘制和保存的训练损失。
               -- self.model_names（str列表）：定义我们训练中使用的网络。
               -- self.visual_names（str列表）：指定要显示和保存的图像。
               -- self.optimizers（优化器列表）：定义并初始化优化器。您可以为每个网络定义一个优化器。如果同时更新两个网络，可以使用itertools.chain将它们分组。参见cycle_gan_model.py的示例。
           """
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # 获取设备名称：CPU或GPU
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # 将所有检查点保存到save_dir
        if opt.preprocess != 'scale_width':  # 使用[scale_width]时，输入图像可能具有不同的尺寸，这会影响cudnn.benchmark的性能。
            torch.backends.cudnn.benchmark = True
        self.loss_names = []  # 训练损失的名称列表
        self.model_names = []  # 使用的网络的名称列表
        self.visual_names = []  # 要显示和保存的图像的名称列表
        self.optimizers = []  # 优化器列表
        self.image_paths = []  # 图像路径列表
        self.metric = 0  # 用于学习率策略'plateau'

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """添加新的模型特定选项，并重新设置现有选项的默认值。

        参数：
            parser          -- 原始选项解析器
            is_train (bool) -- 是否处于训练阶段。您可以使用此标志添加特定于训练或测试的选项。

        返回：
            修改后的选项解析器。
        """
        return parser

    @abstractmethod
    def set_input(self, input):
        """从数据加载器中解包输入数据并执行必要的预处理步骤。

        参数：
            input (dict): 包括数据本身及其元数据信息。
        """
        pass

    @abstractmethod
    def forward(self):
        """运行前向传播；由<optimize_parameters>和<test>函数调用。"""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """计算损失、梯度并更新网络权重；在每个训练迭代中调用。"""
        pass

    def setup(self, opt):
        """加载和打印网络；创建调度器

        参数：
            opt (Option类) -- 存储所有实验标志；需要是BaseOptions的子类
        """
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        if not self.isTrain or opt.continue_train:
            load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
            self.load_networks(load_suffix)
        self.print_networks(opt.verbose)

    def eval(self):
        """在测试时将模型设置为评估模式"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def test(self):
        """在测试时使用的前向传播函数。

        此函数在no_grad()中包装<forward>函数，以便不会保存反向传播的中间步骤。
        它还调用<compute_visuals>以生成附加的可视化结果。
        """
        with torch.no_grad():
            self.forward()
            self.compute_visuals()

    def compute_visuals(self):
        """计算用于visdom和HTML可视化的附加输出图像"""
        pass

    def get_image_paths(self):
        """返回用于加载当前数据的图像路径"""
        return self.image_paths

    def update_learning_rate(self):
        """更新所有网络的学习率；在每个时代结束时调用"""
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('学习率 = %.7f' % lr)

    def get_current_visuals(self):
        """返回可视化图像。train.py将使用visdom显示这些图像，并将图像保存到HTML中"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        """返回训练损失/错误。train.py将在控制台上打印出这些错误，并将它们保存到一个文件中"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...)适用于标量张量和浮点数
        return errors_ret

    def save_networks(self, epoch):
        """将所有网络保存到磁盘。

        参数：
            epoch (int) -- 当前时代；用于文件名'%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """修复InstanceNorm checkpoints不兼容（在0.4版本之前）"""
        key = keys[i]
        if i + 1 == len(keys):  # 在末尾，指向参数/缓冲区
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_networks(self, epoch):
        """从磁盘加载所有网络。

        参数：
            epoch (int) -- 当前时代；用于文件名'%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('from %s loading model' % load_path)
                # 如果您使用的是0.4版本之后的PyTorch（例如，从GitHub源码构建的版本），
                # 您可以删除self.device上的str()。
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # 修复0.4版本之前的InstanceNorm checkpoints
                for key in list(state_dict.keys()):  # 在此处需要复制键，因为我们在循环中改变了
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)

    def print_networks(self, verbose):
        """打印网络中的总参数数量以及（如果verbose）网络体系结构

        参数：
            verbose (bool) -- 如果verbose: 打印网络体系结构
        """
        print('---------- initalizing model -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[网络 %s] 总参数数量 : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        """将所有网络的requires_grad设置为False，以避免不必要的计算

        参数：
            nets (网络列表)   -- 网络列表
            requires_grad (bool)  -- 网络是否需要梯度
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
