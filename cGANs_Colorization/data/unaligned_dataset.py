import os.path
from cGANs_Colorization.data.base_dataset import BaseDataset, get_transform
from cGANs_Colorization.data.image_folder import make_dataset
from PIL import Image
import random


class UnalignedDataset(BaseDataset):
    """
    此数据集类可以加载不对齐/未配对的数据集。

    它需要两个目录来保存来自域A '/path/to/data/trainA' 和来自域B '/path/to/data/trainB' 的训练图像。
    您可以使用数据集标志 '--dataroot /path/to/data' 来训练模型。
    同样，在测试时，您需要准备两个目录：
    '/path/to/data/testA' 和 '/path/to/data/testB'。

    """

    def __init__(self, opt):
        """初始化此数据集类。

        参数:
            opt (Option类) -- 存储所有实验标志的类; 需要是BaseOptions的子类

        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # 创建路径 '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # 创建路径 '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # 从 '/path/to/data/trainA' 加载图像
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # 从 '/path/to/data/trainB' 加载图像
        self.A_size = len(self.A_paths)  # 获取数据集 A 的大小
        self.B_size = len(self.B_paths)  # 获取数据集 B 的大小
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # 获取输入图像的通道数
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # 获取输出图像的通道数
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    def __getitem__(self, index):
        """返回一个数据点及其元数据信息。

        参数:
            index (int)      -- 用于数据索引的随机整数

        返回:
            一个包含 A、B、A_paths 和 B_paths 的字典
                A (张量)       -- 输入域中的图像
                B (张量)       -- 目标域中对应的图像
                A_paths (字符串)    -- 图像路径
                B_paths (字符串)    -- 图像路径
        """
        A_path = self.A_paths[index % self.A_size]  # 确保索引在合理范围内
        if self.opt.serial_batches:   # 确保索引在合理范围内
            index_B = index % self.B_size
        else:   # 随机化域B的索引，以避免固定配对。
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        # 应用图像变换
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """返回数据集中的图像总数。

        由于我们有两个数据集，它们的图像数量可能不同，所以我们取最大值。
        """
        return max(self.A_size, self.B_size)
