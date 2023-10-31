from cGANs_Colorization.data.base_dataset import BaseDataset, get_transform
from cGANs_Colorization.data.image_folder import make_dataset
from PIL import Image

class SingleDataset(BaseDataset):
    """这个数据集类可以加载由路径--dataroot /path/to/data指定的一组图像。

    它可用于仅使用模型选项'-model test'为一个方向生成CycleGAN结果。
    """

    def __init__(self, opt):
        """初始化这个数据集类。

        参数:
            opt (Option类) -- 存储所有实验标志的类; 需要是BaseOptions的子类
        """
        BaseDataset.__init__(self, opt)
        self.A_paths = sorted(make_dataset(opt.dataroot, opt.max_dataset_size))
        input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.transform = get_transform(opt, grayscale=(input_nc == 1))

    def __getitem__(self, index):
        """返回一个数据点和其元数据信息。

        参数:
            index - - 用于数据索引的随机整数

        返回一个包含A和A_paths的字典
            A(tensor) - - 一个领域内的图像
            A_paths(str) - - 图像的路径
        """
        A_path = self.A_paths[index]
        A_img = Image.open(A_path).convert('RGB')
        A = self.transform(A_img)
        return {'A': A, 'A_paths': A_path}

    def __len__(self):
        """返回数据集中的图像总数。"""
        return len(self.A_paths)
