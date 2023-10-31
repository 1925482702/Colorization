import torch
from .base_model import BaseModel
from . import networks


class Pix2PixModel(BaseModel):
    """ 这个类实现了pix2pix模型，用于学习给定配对数据的输入图像到输出图像的映射。

    模型训练需要'--dataset_mode aligned'数据集。
    默认情况下，它使用'--netG unet256' U-Net 生成器，
    一个'--netD basic'鉴别器（PatchGAN），
    和一个'--gan_mode'普通GAN损失（原始GAN论文中使用的交叉熵目标）。

    pix2pix论文: https://arxiv.org/pdf/1611.07004.pdf
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """添加新的特定于数据集的选项，并重写现有选项的默认值。

        参数：
            parser          -- 原始的选项解析器
            is_train (bool) -- 是否为训练阶段。您可以使用此标志添加特定于训练或特定于测试的选项。

        返回：
            修改后的解析器。

        对于pix2pix，我们不使用图像缓冲区。
        训练目标是：GAN损失 + lambda_L1 * ||G(A)-B||_1
        默认情况下，我们使用普通GAN损失，带有批归一化的UNet，以及对齐的数据集。
        """
        # 将默认值更改为匹配pix2pix论文（https://phillipi.github.io/pix2pix/）
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')  # 池大小，GAN模式
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='L1损失的权重')

        return parser

    def __init__(self, opt):
        """初始化pix2pix类。

        参数：
            opt (Option class)-- 存储所有实验标志的对象；需要是BaseOptions的子类
        """
        BaseModel.__init__(self, opt)
        # 指定要打印的训练损失。训练/测试脚本将调用<BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        # 指定要保存/显示的图像。训练/测试脚本将调用<BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # 指定要保存到磁盘的模型。训练/测试脚本将调用<BaseModel.save_networks>和<BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # 测试时，只加载G
            self.model_names = ['G']

        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """从数据加载器中解压缩输入数据并执行必要的预处理步骤。

        参数：
            input (dict): 包括数据本身和其元数据信息。

        选项'direction'可用于交换A域和B域中的图像。
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """运行前向传播；由<optimize_parameters>和<test>函数调用。"""
        self.fake_B = self.netG(self.real_A)

    def backward_D(self):
        """计算鉴别器的GAN损失"""
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5 # 损失取平均
        self.loss_D.backward()

    def backward_G(self):
        """计算生成器的GAN和L1损失"""
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
