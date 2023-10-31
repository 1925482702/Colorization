"""数据集类模板

该模块提供了一个模板，供用户实现自定义数据集。
您可以指定'--dataset_mode template'来使用此数据集。
类名应与文件名和其dataset_mode选项保持一致。
文件名应为<dataset_mode>_dataset.py
类名应为<Dataset_mode>Dataset.py
您需要实现以下功能：
    -- <modify_commandline_options>: 添加特定于数据集的选项并重写现有选项的默认值。
    -- <__init__>: 初始化此数据集类。
    -- <__getitem__>: 返回数据点及其元数据信息。
    -- <__len__>: 返回图像的数量。
"""
from cGANs_Colorization.data.base_dataset import BaseDataset, get_transform
# from data.image_folder import make_dataset
# from PIL import Image


class TemplateDataset(BaseDataset):
    """一个模板数据集类，供您实现自定义数据集。"""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """添加新的特定于数据集的选项，并重写现有选项的默认值。

        参数:
            parser          -- 原始选项解析器
            is_train (bool) -- 是否为训练阶段。您可以使用此标志添加特定于训练或测试的选项。

        返回:
            修改后的解析器。
        """
        parser.add_argument('--new_dataset_option', type=float, default=1.0, help='新的数据集选项')
        parser.set_defaults(max_dataset_size=10, new_dataset_option=2.0)  # 指定特定于数据集的默认值
        return parser

    def __init__(self, opt):
        """初始化此数据集类。

        参数:
            opt (Option类) -- 存储所有实验标志的类; 需要是BaseOptions的子类

        这里可以做一些事情。
        - 保存选项（已在BaseDataset中完成）
        - 获取数据集的图像路径和元信息。
        - 定义图像转换。
        """
        # 保存选项和数据集根目录
        BaseDataset.__init__(self, opt)
        # 获取数据集的图像路径；
        self.image_paths = []  # 您可以调用sorted(make_dataset(self.root, opt.max_dataset_size))以获取目录self.root下的所有图像路径
        # 定义默认的变换函数。您可以使用<base_dataset.get_transform>；您也可以定义自己的自定义变换函数
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        """返回一个数据点及其元数据信息。

        参数:
            index -- 用于数据索引的随机整数

        返回:
            一个包含其名称的数据字典。通常包含数据本身及其元数据信息。

        步骤1：获取一个随机图像路径，例如，path = self.image_paths[index]
        步骤2：从磁盘加载您的数据，例如，image = Image.open(path).convert('RGB')。
        步骤3：将您的数据转换为PyTorch张量。您可以使用诸如self.transform之类的帮助程序函数。例如，data = self.transform(image)
        步骤4：返回一个数据点作为字典。
        """
        path = 'temp'    # 需要是一个字符串
        data_A = None    # 需要是一个张量
        data_B = None    # 需要是一个张量
        return {'data_A': data_A, 'data_B': data_B, 'path': path}

    def __len__(self):
        """返回图像的总数。"""
        return len(self.image_paths)
