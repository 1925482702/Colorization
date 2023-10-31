import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler


###############################################################################
# Helper Functions
###############################################################################

class Identity(nn.Module):
    def forward(self, x):
        return x

## 标准化层
def get_norm_layer(norm_type='instance'):
    """返回一个归一化层

    参数:
        norm_type (str) -- 归一化层的名称: batch | instance | none

    对于BatchNorm，我们使用可学习的仿射参数并跟踪运行时统计信息（均值/标准差）。
    对于InstanceNorm，我们不使用可学习的仿射参数。我们不跟踪运行时统计信息。
    """

    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

## 学习率策略
def get_scheduler(optimizer, opt):
    """返回一个学习率调度器

    参数:
        optimizer          -- 网络的优化器
        opt (option 类)    -- 存储所有实验标志的类; 需要是 BaseOptions 的子类．
                              opt.lr_policy 是学习率策略的名称: linear | step | plateau | cosine

    对于 'linear'，我们在前 <opt.n_epochs> 个时期保持相同的学习率，
    然后在接下来的 <opt.n_epochs_decay> 个时期线性地将速率减少到零。
    对于其他调度器（step，plateau 和 cosine），我们使用默认的 PyTorch 调度器。
    有关更多详细信息，请参阅 https://pytorch.org/docs/stable/optim.html。
    """

    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

## 权重初始化
def init_weights(net, init_type='normal', init_gain=0.02):
    """初始化网络权重。

    参数:
        net (网络)        -- 要初始化的网络
        init_type (str)  -- 初始化方法的名称: normal | xavier | kaiming | orthogonal
        init_gain (float) -- normal、xavier 和 orthogonal 的缩放因子。

    在原始的 pix2pix 和 CycleGAN 论文中使用 'normal'。但是对于某些应用，xavier 和 kaiming 可能效果更好。
    可以随时自行尝试。
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

## 模型初始化
def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """初始化网络：1. 注册CPU/GPU设备（支持多GPU）；2. 初始化网络权重

    参数:
        net (网络)          -- 要初始化的网络
        init_type (str)    -- 初始化方法的名称: normal | xavier | kaiming | orthogonal
        gain (float)       -- normal、xavier 和 orthogonal 的缩放因子。
        gpu_ids (int list) -- 网络运行在哪些GPU上：例如，0,1,2

    返回一个初始化后的网络。
    """

    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net

## 生成器定义
def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    # 创建生成器
    # 参数：
    # input_nc（int）——输入图像中的通道数
    # output_nc（int）—输出图像中的通道数
    # ngf（int）—最后一个conv层中的筛选器数
    # netG（str）--体系结构的名称：resnet_9blocks | resnet_6blocks | unet_256 | unet_128
    # norm（str）--网络中使用的规范化层的名称：batch | instance | none
    # use_dropout（bool）--如果使用丢弃层。
    # init_type（str）——初始化方法的名称。
    # init_gain（float）——法线、xavier和正交的比例因子。
    # gpu_ids（int
    # list）——网络运行在哪些gpu上：例如，0, 1, 2
    # 返回生成器
    # 我们目前的实现提供了两种类型的生成器：
    # U - Net: [unet_128]（用于128x128输入图像）和[unet_256]（用于256x256输入图像）
    # U - Net的原始论文：https: // arxiv.org / abs / 1505.04597
    # 基于Resnet的生成器：[Resnet_6blocks]（具有6个Resnet块）和[Resnet_9blocks]
    # 基于Resnet的生成器由几个下采样 / 上采样操作之间的几个Resnet块组成。
    # 我们改编了Justin
    # Johnson的神经风格转移项目中的Torch代码(https: // github.com / jcjohnson / fast - neural - style)。
    # 生成器已由 < init_net > 初始化。它将RELU用于非线性。

    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)

## 判别器定义
def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """创建一个鉴别器

    参数:
        input_nc (int)     -- 输入图像的通道数
        ndf (int)          -- 第一个卷积层中的滤波器数目
        netD (str)         -- 架构的名称: basic | n_layers | pixel
        n_layers_D (int)   -- 鉴别器中的卷积层数; 仅在 netD=='n_layers' 时有效
        norm (str)         -- 网络中使用的归一化层的类型。
        init_type (str)    -- 初始化方法的名称。
        init_gain (float)  -- normal、xavier 和 orthogonal 的缩放因子。
        gpu_ids (int list) -- 网络运行在哪些GPU上：例如，0,1,2

    返回一个鉴别器

    我们当前的实现提供了三种类型的鉴别器:
        [basic]: 在原始 pix2pix 论文中描述的 'PatchGAN' 分类器。
        它可以分类 70×70 的重叠补丁是真实的还是假的。
        这样一个基于补丁级别的鉴别器架构比全图鉴别器具有更少的参数
        并且可以在完全卷积的方式下处理任意大小的图像。

        [n_layers]: 使用此模式，您可以通过参数 <n_layers_D> 指定鉴别器中的卷积层数
        （默认值为3，就像在 [basic]（PatchGAN）中使用的一样）。

        [pixel]: 1x1 的 PixelGAN 鉴别器可以分类一个像素是真实的还是假的。
        它鼓励更大的颜色多样性，但对空间统计没有影响。

    鉴别器已经由 <init_net> 进行了初始化。它使用泄漏整流线性单元（Leaky RELU）进行非线性激活。
    """

    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """初始化 GANLoss 类。

        参数:
            gan_mode (str) - - GAN 目标的类型。目前支持 vanilla、lsgan 和 wgangp。
            target_real_label (bool) - - 真实图像的标签
            target_fake_label (bool) - - 伪造图像的标签

        注意: 不要在鉴别器的最后一层使用 sigmoid 函数。
        LSGAN 不需要 sigmoid 函数。vanilla GANs 将使用 BCEWithLogitsLoss 来处理它。
        """

        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """创建与输入相同大小的标签张量。

        参数:
            prediction (tensor) - - 通常是来自鉴别器的预测
            target_is_real (bool) - - 地面实况标签是真实图像还是伪造图像

        返回:
            一个填充有地面实况标签的标签张量，与输入的大小相同
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """根据鉴别器的输出和地面实况标签计算损失。

        参数:
            prediction (tensor) - - 通常是来自鉴别器的预测输出
            target_is_real (bool) - - 地面实况标签是真实图像还是伪造图像

        返回:
            计算得到的损失值。
        """

        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

## 计算梯度惩罚
def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """计算梯度惩罚损失，用于WGAN-GP论文 https://arxiv.org/abs/1704.00028

    参数:
        netD (网络)                  -- 鉴别器网络
        real_data (张量数组)         -- 真实图像
        fake_data (张量数组)         -- 生成器生成的图像
        device (str)                -- GPU / CPU: 使用 torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- 是否混合使用真实和伪造数据 [real | fake | mixed].
        constant (float)            -- 公式中使用的常数 ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- 此损失的权重

    返回梯度惩罚损失
    """

    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None

## ResNet生成器
class ResnetGenerator(nn.Module):
    """基于ResNet的生成器，它由几个ResNet块组成，这些块位于一些下采样/上采样操作之间。

    我们参考了Justin Johnson的神经风格转移项目(https://github.com/jcjohnson/fast-neural-style)中的Torch代码和思路。
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """构建基于ResNet的生成器

        参数:
            input_nc (int)      -- 输入图像的通道数
            output_nc (int)     -- 输出图像的通道数
            ngf (int)           -- 最后一个卷积层中的滤波器数量
            norm_layer          -- 规范化层
            use_dropout (bool)  -- 是否使用dropout层
            n_blocks (int)      -- ResNet块的数量
            padding_type (str)  -- 卷积层中填充层的名称: reflect | replicate | zero
        """

        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)

## ResNet模块
class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """初始化Resnet块

        一个Resnet块是一个带有跳跃连接的卷积块
        我们使用build_conv_block函数构建一个卷积块，并在forward函数中实现跳跃连接。
        原始的Resnet论文: https://arxiv.org/pdf/1512.03385.pdf
        """

        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """构建一个卷积块。

        参数:
            dim (int)            -- 卷积层中的通道数。
            padding_type (str)   -- 填充层的名称: reflect | replicate | zero
            norm_layer           -- 标准化层
            use_dropout (bool)   -- 是否使用dropout层。
            use_bias (bool)      -- 卷积层是否使用偏置项

        返回一个卷积块（包含一个卷积层，一个标准化层和一个非线性层（ReLU））
        """

        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

## UNet生成器结构
class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""
# num_downs 下采样次数
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """构建一个Unet生成器
        参数:
            input_nc (int)  -- 输入图像中的通道数
            output_nc (int) -- 输出图像中的通道数
            num_downs (int) -- UNet中的下采样次数。例如，如果|num_downs| == 7，
                               大小为128x128的图像将会在瓶颈处变为1x1的大小
            ngf (int)       -- 最后一个卷积层中的滤波器数量
            norm_layer      -- 标准化层

        我们从最内层到最外层构建U-Net。
        这是一个递归过程。
        """

        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  ## 最内层跳层连接
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            # submodule=unet_block说明unet内部还有层次
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  ## 最外层跳层连接

    def forward(self, input):
        """Standard forward"""
        return self.model(input)

## UNet跳层连接模块
class UnetSkipConnectionBlock(nn.Module):
    # outer_nc输出通道数，inner_nc输入通道数
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """构建一个带有跳跃连接的Unet子模块。

        参数:
            outer_nc (int) -- 外部卷积层中的滤波器数量
            inner_nc (int) -- 内部卷积层中的滤波器数量
            input_nc (int) -- 输入图像/特征中的通道数
            submodule (UnetSkipConnectionBlock) -- 之前定义的子模块
            outermost (bool)    -- 如果此模块是最外层模块
            innermost (bool)    -- 如果此模块是最内层模块
            norm_layer          -- 标准化层
            use_dropout (bool)  -- 是否使用dropout层
        """

        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc

        ## 下采样层
        #卷积大小为4，步长为2
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        ## 上采样层
        if outermost: ##最外层，有特征拼接
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv] ##下采样模块
            up = [uprelu, upconv, nn.Tanh()] ##上采样模块
            model = down + [submodule] + up
        elif innermost: ## 最内层，无特征拼接
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else: ## 中间层，有特征拼接
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            #下采样激活函数，下采样卷积，下采样标准化层
            down = [downrelu, downconv, downnorm]
            # 上采样激活函数，上采样卷积，上采样标准化层
            up = [uprelu, upconv, upnorm]
            # 只有在最内层没有特征拼接
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)

## PatchGAN，basic PatchGAN全局步长=16，感受野大小为70*70
class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """构建一个PatchGAN鉴别器

        参数:
            input_nc (int)  -- 输入图像中的通道数
            ndf (int)       -- 最后一个卷积层中的滤波器数量
            n_layers (int)  -- 鉴别器中的卷积层数量
            norm_layer      -- 标准化层
        """

        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        ## 4 * 4 卷积，步长为2
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        ## 4 * 4 卷积，步长为2
        for n in range(1, n_layers):  ## 逐渐增加输出通道数，默认是3个卷积层
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        ## 4 * 4 卷积，步长为1
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        ## 4 * 4 卷积，步长为1， 输出单通道预测概率图
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

## 1x1 PatchGAN结构
class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """构建一个1x1 PatchGAN鉴别器

        参数:
            input_nc (int)  -- 输入图像中的通道数
            ndf (int)       -- 最后一个卷积层中的滤波器数量
            norm_layer      -- 标准化层
        """

        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)
