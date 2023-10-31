
import torch
import torch.nn as nn
import numpy as np
from IPython import embed
from .base_color import *

class CNNGenerator(BaseColor):
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(CNNGenerator, self).__init__()

        # 创建一个二维卷积层，输入通道数为1（因为处理的是灰度图像），输出通道数为64，卷积核大小为3x3，步幅为1，填充为1，使用偏置项。
        # 整行代码用列表括起来，因为这是model1的第一层。
        model1=[nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True),]

        # 添加一个ReLU激活函数层，True表示inplace模式，即直接在原地修改输入，节省内存。
        model1+=[nn.ReLU(True),]

        # 类似于第一层，这里再添加了一个卷积层，将输入通道数从64变为64，卷积核大小为3x3，步幅为2，填充为1。
        # 这行代码将第二个卷积层添加到model1的第三层
        model1+=[nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),]

        # 添加一个ReLU激活函数层，True表示inplace模式，即直接在原地修改输入，节省内存。
        model1+=[nn.ReLU(True),]

        #添加一个批归一化（Batch Normalization）层，输入通道数为64。
        # 批归一化（Batch
        # Normalization）是深度学习中一种用于加速神经网络训练过程的技术。
        # 它的基本思想是在神经网络的每一层的输入数据上进行归一化处理，使得数据的分布在训练过程中保持稳定。
        # 具体来说，对于一个神经网络的某一层，批归一化会计算出该层输入数据的均值和方差，
        # 并将数据进行线性变换和平移，使得均值为0，方差为1。这样做的好处是可以防止网络在训练过程中出现梯度消失或梯度爆炸的问题，从而加速训练的收敛过程。
        model1+=[norm_layer(64),]




        model2=[nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),]
        model2+=[nn.ReLU(True),]
        model2+=[nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),]
        model2+=[nn.ReLU(True),]
        model2+=[norm_layer(128),]

        model3=[nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model3+=[nn.ReLU(True),]
        model3+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model3+=[nn.ReLU(True),]
        model3+=[nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),]
        model3+=[nn.ReLU(True),]
        model3+=[norm_layer(256),]

        model4=[nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model4+=[nn.ReLU(True),]
        model4+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model4+=[nn.ReLU(True),]
        model4+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model4+=[nn.ReLU(True),]
        model4+=[norm_layer(512),]

        model5=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model5+=[nn.ReLU(True),]
        model5+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model5+=[nn.ReLU(True),]
        model5+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model5+=[nn.ReLU(True),]
        model5+=[norm_layer(512),]

        model6=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model6+=[nn.ReLU(True),]
        model6+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model6+=[nn.ReLU(True),]
        model6+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model6+=[nn.ReLU(True),]
        model6+=[norm_layer(512),]

        model7=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model7+=[nn.ReLU(True),]
        model7+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model7+=[nn.ReLU(True),]
        model7+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model7+=[nn.ReLU(True),]
        model7+=[norm_layer(512),]

        # 从这里开始进行反卷积操作ConvTranspose2d

        # 这是一个反卷积层（也称为转置卷积或上采样层）。它将输入特征图从512通道转置成256通道。
        # 表示卷积核的大小为4x4。
        # 步长 = 2，这会使得输出特征图的尺寸是输入的两倍。
        #    这是因为在反卷积操作中，每个步幅会在输出特征图的两个相邻像素之间插入一个新的像素。
        #    这样，随着卷积核的滑动，输出特征图的尺寸会相应地扩大。
        #    在这种情况下，选择步幅为2可能是为了实现上采样的效果，从而将特征图的尺寸扩大，同时保留重要的特征信息。
        #    这在某些任务中（如图像生成或分割）是非常有用的。

        # padding = 1，表示在输入的周围填充1个像素，以保持尺寸一致。
        # bias = True表示该层会学习一个偏置项。
        model8=[nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True),]

        model8+=[nn.ReLU(True),]
        model8+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model8+=[nn.ReLU(True),]
        model8+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model8+=[nn.ReLU(True),]

        # 将输入的特征图从 256 个通道变换为 313 个通道
        # 在这里，网络设计为最后一层输出313个通道，这是因为作者认为313个通道足以覆盖绝大部分颜色的可能性。
        # 这样，网络就能够学会将灰度图像映射到CIE Lab 颜色空间的a和b通道上，从而实现彩色化的效果。
        model8+=[nn.Conv2d(256, 313, kernel_size=1, stride=1, padding=0, bias=True),]

        # 每个 self.modelX 都是一个包含了一系列卷积层、ReLU激活函数以及批归一化层的序列。这种组织方式可以使得模型的结构更清晰，也更方便进行前向传播。
        # 在模型的前向传播过程中，可以通过调用self.modelX(input_l) 来依次经过相应的模块，从而完成整个网络的前向计算。
        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)
        self.model8 = nn.Sequential(*model8)

        self.softmax = nn.Softmax(dim=1)
        # 这个层在模型的最后用于将网络输出的特征图转化为颜色信息。
        self.model_out = nn.Conv2d(313, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=False)
        # 这是一个上采样层，它将输入的特征图在两个维度上分别放大4倍，使用的上采样方式是双线性插值。这个层的作用是将特征图的尺寸放大，以便与输入图像的尺寸相匹配。
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')


    # 前向传播函数
    # input_l是输入的L通道图像，经过归一化处理。
    # conv1_2 = self.model1(self.normalize_l(input_l))表示将输入的L通道图像经过第一个模型(model1)处理，得到的结果保存在conv1_2中。
    # 然后，依次将conv1_2输入到model2, model3, ..., model8中进行处理，得到conv2_2, conv3_3, ..., conv8_3。
    # out_reg = self.model_out(self.softmax(conv8_3))表示将经过最后一个模型model8处理后的特征图conv8_3输入到model_out中，经过softmax处理得到输出out_reg。
    # 最后，将输出out_reg经过上采样操作self.upsample4进行放大，然后通过self.unnormalize_ab进行反归一化处理，得到最终的ab通道图像
    def forward(self, input_l):
        conv1_2 = self.model1(self.normalize_l(input_l))
        conv2_2 = self.model2(conv1_2)
        conv3_3 = self.model3(conv2_2)
        conv4_3 = self.model4(conv3_3)
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)
        conv8_3 = self.model8(conv7_3)
        out_reg = self.model_out(self.softmax(conv8_3))

        return self.unnormalize_ab(self.upsample4(out_reg))

def CNN(pretrained=True):
	model = CNNGenerator()
	if(pretrained):
		import torch.utils.model_zoo as model_zoo
		model.load_state_dict(model_zoo.load_url('https://colorizers.s3.us-east-2.amazonaws.com/colorization_release_v2-9b330a0b.pth',map_location='cpu',check_hash=True))
	return model
