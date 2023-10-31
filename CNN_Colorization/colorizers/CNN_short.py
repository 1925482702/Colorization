import torch
import torch.nn as nn
from .base_color import *
from .util import *
class CNN_ShortGenerator(BaseColor):
    def __init__(self, norm_layer=nn.BatchNorm2d, classes=529):
        super(CNN_ShortGenerator, self).__init__()

        # Conv1
        # 接受输入通道数为 4，输出通道数为 64，卷积核大小为 3x3，步长为 1，填充为 1
        model1=[nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1, bias=True),]
        # relu激活函数将所有负值置为零，保持正值不变
        model1+=[nn.ReLU(True),]
        model1+=[nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),]
        model1+=[nn.ReLU(True),]
        # 批归一化层，它将对输入进行归一化处理，以加速训练过程并提高模型的稳定性
        model1+=[norm_layer(64),]
        # add a subsampling operation

        # Conv2
        model2=[nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),]
        model2+=[nn.ReLU(True),]
        model2+=[nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),]
        model2+=[nn.ReLU(True),]
        model2+=[norm_layer(128),]
        # add a subsampling layer operation

        # Conv3
        model3=[nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model3+=[nn.ReLU(True),]
        model3+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model3+=[nn.ReLU(True),]
        model3+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model3+=[nn.ReLU(True),]
        model3+=[norm_layer(256),]
        # add a subsampling layer operation

        # Conv4
        model4=[nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model4+=[nn.ReLU(True),]
        model4+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model4+=[nn.ReLU(True),]
        model4+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model4+=[nn.ReLU(True),]
        model4+=[norm_layer(512),]

        # Conv5
        model5=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model5+=[nn.ReLU(True),]
        model5+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model5+=[nn.ReLU(True),]
        model5+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model5+=[nn.ReLU(True),]
        model5+=[norm_layer(512),]

        # Conv6
        model6=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model6+=[nn.ReLU(True),]
        model6+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model6+=[nn.ReLU(True),]
        model6+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model6+=[nn.ReLU(True),]
        model6+=[norm_layer(512),]

        # Conv7
        model7=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model7+=[nn.ReLU(True),]
        model7+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model7+=[nn.ReLU(True),]
        model7+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model7+=[nn.ReLU(True),]
        model7+=[norm_layer(512),]
        # 多个卷积层的堆叠可以逐步提取图像的抽象特征，从简单的边缘和纹理到更高级别的特征，如形状、纹理组合等

        # 反卷积
        # Conv7
        # 首先进行了一个反卷积操作，然后接着进行了一系列卷积操作。
        # 这是因为在神经网络中，反卷积操作（也称为转置卷积或上采样操作）通常用于将特征图的分辨率扩大，从而使得模型可以捕获更细节的信息。
        # 然而，反卷积可能会引入一些不必要的高频噪声或者过度扩大特征图。为了保持特征图的质量和稳定性，常常会在反卷积后接上一些正常的卷积层。
        model8up=[nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True)]


        # 这里采用了短路链接
        # 短路连接（Short - Circuit
        # Connection），也被称为跳跃连接（Skip
        # Connection）或恒等映射（Identity
        # Mapping），是指在神经网络中将某一层（或某些层）的输出直接添加到后续层的输入中，形成了一个直接的连接路径。
        #
        # 这样的设计通常出现在残差网络（Residual
        # Network，ResNet）等架构中。短路连接的目的是解决深度神经网络训练过程中出现的梯度消失或梯度爆炸的问题，从而使得网络能够更深更容易训练。
        #
        # 具体来说，短路连接允许梯度在反向传播时直接传递到更早的层，避免了在深度网络中逐层传递梯度时可能发生的消失或爆炸。
        #
        # 短路连接的另一个重要特性是，它允许网络学习某些恒等映射，即将输入直接传递到输出，这也使得网络可以选择性地学习对输入的修正。
        #
        # 总的来说，短路连接是一种有助于解决深度神经网络训练问题的重要技术，它使得设计更深的网络变得可行，并且在实践中取得了显著的成功。
        model3short8=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]

        model8=[nn.ReLU(True),]
        model8+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model8+=[nn.ReLU(True),]
        model8+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model8+=[nn.ReLU(True),]
        model8+=[norm_layer(256),]

        # Conv9
        model9up=[nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=True),]
        model2short9=[nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),]
        # add the two feature maps above        

        model9=[nn.ReLU(True),]
        model9+=[nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),]
        model9+=[nn.ReLU(True),]
        model9+=[norm_layer(128),]

        # Conv10
        model10up=[nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=True),]
        model1short10=[nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),]


        # add the two feature maps above

        model10=[nn.ReLU(True),]
        model10+=[nn.Conv2d(128, 128, kernel_size=3, dilation=1, stride=1, padding=1, bias=True),]
        # 带有负斜率的 Leaky ReLU 激活函数，它在输入为负时不再取零，而是乘以一个小的负斜率（这里设定为0.2）
        # 有助于模型学习更复杂的特征表示，并引入了非线性以提高模型的表达能力
        model10+=[nn.LeakyReLU(negative_slope=.2),]


        # 这是一个卷积层，输入通道数为256，输出通道数为classes（在模型初始化时传入的参数）。
        # 这个卷积层的作用是将从模型中提取的特征进行分类，输出一个与类别数相等的通道数
        # 使用256个通道可能是作者根据实验证明在这里使用较低的通道数可以获得更好的结果。
        # 这也显示了神经网络设计中的许多选择是经验性的，通常需要通过实验来找到最佳的配置
        # classification output
        model_class=[nn.Conv2d(256, classes, kernel_size=1, padding=0, dilation=1, stride=1, bias=True),]

        # 这个卷积层的作用是对模型提取的特征进行回归，输出一个包含两个通道的特征图
        # regression output
        model_out=[nn.Conv2d(128, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=True),]
        # 经过了 nn.Tanh() 操作，它将输出的值压缩到 -1 到 1 的范围内。这通常在回归任务中使用，以保证输出的范围在合适的区间内
        model_out+=[nn.Tanh()]

        # self.model1, self.model2, self.model3, ... 直到 self.model10
        # 分别是一系列的卷积层和激活函数，它们在模型的前向传播中用于提取图像特征。
        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)

        #self.model8up, self.model9up, self.model10up
        # 分别是反卷积层，用于将特征图的分辨率扩大，从而使得模型可以捕获更细节的信息。
        self.model8up = nn.Sequential(*model8up)
        self.model8 = nn.Sequential(*model8)
        self.model9up = nn.Sequential(*model9up)
        self.model9 = nn.Sequential(*model9)
        self.model10up = nn.Sequential(*model10up)
        self.model10 = nn.Sequential(*model10)


        # self.model3short8, self.model2short9, self.model1short10
        # 是用于短路连接的卷积层，它们将较浅层的特征图与相应深层的特征图相结合。
        self.model3short8 = nn.Sequential(*model3short8)
        self.model2short9 = nn.Sequential(*model2short9)
        self.model1short10 = nn.Sequential(*model1short10)

        # self.model_class 和 self.model_out 分别是用于分类和回归的卷积层。
        self.model_class = nn.Sequential(*model_class)
        self.model_out = nn.Sequential(*model_out)

        # self.upsample4 是一个上采样层，用于将特征图的尺寸放大4倍。
        self.upsample4 = nn.Sequential(*[nn.Upsample(scale_factor=4, mode='bilinear'),])
        # self.softmax 是一个用于分类任务的 softmax 操作
        self.softmax = nn.Sequential(*[nn.Softmax(dim=1),])

    def forward(self, input_A, input_B=None, mask_B=None):
        # 如果input_B为空，将其初始化为与input_A形状相同的全零张量。
        # 如果mask_B为空，将其初始化为与input_A形状相同的全零张量。
        if(input_B is None):
            input_B = torch.cat((input_A*0, input_A*0), dim=1)
        if(mask_B is None):
            mask_B = input_A*0

        #将 input_A、经过归一化的 input_B、以及 mask_B 拼接在一起，然后通过 model1 进行卷积。
        conv1_2 = self.model1(torch.cat((self.normalize_l(input_A),self.normalize_ab(input_B),mask_B),dim=1))

        # 将第一步得到的特征图经过一系列的卷积、下采样操作，得到conv2_2, conv3_3, conv4_3, conv5_3, conv6_3, conv7_3
        # 分别对应模型中的Conv2, Conv3, Conv4, Conv5, Conv6, Conv7
        conv2_2 = self.model2(conv1_2[:,:,::2,::2])
        conv3_3 = self.model3(conv2_2[:,:,::2,::2])
        conv4_3 = self.model4(conv3_3[:,:,::2,::2])
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)

        # 进行短路连接操作，将conv3_3通过model3short8送入model8up进行反卷积操作，得到conv8_up，然后将其与conv8_3相加，得到conv8_3。
        conv8_up = self.model8up(conv7_3) + self.model3short8(conv3_3)
        conv8_3 = self.model8(conv8_up)
        # 经过一系列类似的操作，得到 conv9_3 和 conv10_2。
        conv9_up = self.model9up(conv8_3) + self.model2short9(conv2_2)
        conv9_3 = self.model9(conv9_up)
        conv10_up = self.model10up(conv9_3) + self.model1short10(conv1_2)
        conv10_2 = self.model10(conv10_up)
        # 将 conv10_2 经过 model_out 进行回归预测，得到 out_reg。
        out_reg = self.model_out(conv10_2)

        # 返回经过反归一化的out_reg
        return self.unnormalize_ab(out_reg)



def CNN_Short(pretrained=True):
    model = CNN_ShortGenerator()
    if(pretrained):
        import torch.utils.model_zoo as model_zoo
        model.load_state_dict(model_zoo.load_url('https://colorizers.s3.us-east-2.amazonaws.com/siggraph17-df00044c.pth',map_location='cpu',check_hash=True))
    return model

