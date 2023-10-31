import os.path
from cGANs_Colorization.data.base_dataset import BaseDataset, get_params, get_transform
from cGANs_Colorization.data.image_folder import make_dataset
from PIL import Image

class AlignedDataset(BaseDataset):
    """配对图像数据集的数据集类。

    假设目录'/path/to/data/train'包含形式为{A,B}的图像对。
    在测试时，您需要准备一个目录'/path/to/data/test'。
    """

    def __init__(self, opt):
        """初始化此数据集类。

        参数：
            opt (Option class) - 存储所有实验标志的类；需要是BaseOptions的子类
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # 获取图像目录
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # 获取图像路径
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size应该小于加载图像的尺寸
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """返回数据点及其元数据信息。

        参数：
            index - - 用于数据索引的随机整数

        返回一个包含A、B、A_paths和B_paths的字典
            A (tensor) - - 输入域中的图像
            B (tensor) - - 目标域中对应的图像
            A_paths (str) - - 图像路径
            B_paths (str) - - 图像路径 (与A_paths相同)
        """
        # 根据随机整数索引读取图像
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        # 将AB图像拆分为A和B
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))

        # 对A和B应用相同的变换
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """返回数据集中的图像总数。"""
        return len(self.AB_paths)
