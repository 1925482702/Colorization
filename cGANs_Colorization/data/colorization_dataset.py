#coding:utf8
import os.path
from cGANs_Colorization.data.base_dataset import BaseDataset, get_transform
from cGANs_Colorization.data.image_folder import make_dataset
from skimage import color  # require skimage
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

## 图像上色任务数据集定义，读取RGB图像，并将其转换为（L，ab）类型数据
class ColorizationDataset(BaseDataset):
    # 该数据集类可以加载一组RGB格式的自然图像，并在Lab颜色空间中将RGB格式转换为（L，ab）对。
    # 此数据集是基于pix2pix的着色模型（“--模型着色”）所必需的

    @staticmethod
    def modify_commandline_options(parser, is_train):
        # 添加新的特定于数据集的选项，并重写现有选项的默认值。
        # 参数：
        # parser—原始选项解析器
        # is_train（bool）——无论是训练阶段还是测试阶段。您可以使用此标志添加特定于培训或特定于测试的选项。
        # 返回：
        # 修改后的解析器。
        # 输入图像通道书为1（L通道），输出通道数为2（ab通道），
        # 方向是A到B
        parser.set_defaults(input_nc=1, output_nc=2, direction='AtoB')
        return parser

    def __init__(self, opt):
        # 初始化此数据集类。
        # 参数：
        # opt（Option类）——存储所有 experiment flags；需要是BaseOptions的子类

        BaseDataset.__init__(self, opt)
        self.dir = os.path.join(opt.dataroot, opt.phase) #取数据集
        self.AB_paths = sorted(make_dataset(self.dir, opt.max_dataset_size))#调用类make_dataset，获取路径AB_path
        assert(opt.input_nc == 1 and opt.output_nc == 2 and opt.direction == 'AtoB')#输入通道是1，输出通道是2（ab），方向为从灰度图变为彩色图
        self.transform = get_transform(self.opt, convert=False)#读取一个图像预处理的函数

    def __getitem__(self, index):
        """
            返回包含a、B、a_path和B_path的字典
            A (tensor) --图像的L通道
            B (tensor) --同一图像的ab通道
            A_paths（str）--映像路径
            B_paths（str）--映像路径（与A_paths相同）
        """
        path = self.AB_paths[index]#根据索引获得图片地址
        im = Image.open(path).convert('RGB') ##读取RGB图
        im = self.transform(im)#预处理操作
        im = np.array(im)
        lab = color.rgb2lab(im).astype(np.float32) ##color来自于skimage包，将RGB图转换为CIELab，L通道值在0～100之间，AB通道值在0～110之间
        lab_t = transforms.ToTensor()(lab)#转成tensor格式
        A = lab_t[[0], ...] / 50.0 - 1.0 ##将L通道(index=0)归一化到-1和1之间
        B = lab_t[[1, 2], ...] / 110.0 ##将A，B通道(index=1,2)归一化到0和1之间
        return {'A': A, 'B': B, 'A_paths': path, 'B_paths': path}

    def __len__(self):
        """返回数据集中图片的数量."""
        return len(self.AB_paths)
