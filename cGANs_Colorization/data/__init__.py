# data/__init__.py

"""这个包包含了所有与数据加载和预处理相关的模块。

要添加一个名为'dummy'的自定义数据集类，您需要添加一个名为'dummy_dataset.py'的文件，并定义一个继承自BaseDataset的子类'DummyDataset'。
您需要实现四个函数：
    -- <__init__>:                      初始化类，首先调用BaseDataset.__init__(self, opt)。
    -- <__len__>:                       返回数据集的大小。
    -- <__getitem__>:                   从数据加载器中获取一个数据点。
    -- <modify_commandline_options>:    （可选）添加特定于数据集的选项并设置默认选项。

现在您可以通过指定标志'--dataset_mode dummy'来使用数据集类。
有关更多详细信息，请参见我们的模板数据集类'template_dataset.py'。
"""

import importlib
import torch.utils.data
from cGANs_Colorization.data.base_dataset import BaseDataset

def find_dataset_using_name(dataset_name):
    """导入模块"data/[dataset_name]_dataset.py"。

    在文件中，将实例化名为DatasetNameDataset()的类。
    它必须是BaseDataset的子类，并且对大小写不敏感。
    """
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("在 %s.py 中，应该有一个名为 %s 的BaseDataset子类，其类名小写后与 %s 匹配。" % (dataset_filename, target_dataset_name, target_dataset_name))

    return dataset

def get_option_setter(dataset_name):
    """返回数据集类的静态方法 <modify_commandline_options>。"""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options

def create_dataset(opt):
    """根据选项创建一个数据集。

    此函数包装了CustomDatasetDataLoader类。
        这是本包与'train.py'/'test.py'之间的主要接口

    示例:
        >>> from cGANs_Colorization.data import create_dataset
        >>> dataset = create_dataset(opt)
    """
    data_loader = CustomDatasetDataLoader(opt)
    dataset = data_loader.load_data()
    return dataset

class CustomDatasetDataLoader():
    """执行多线程数据加载的Dataset类的包装器"""

    def __init__(self, opt):
        """初始化此类

        步骤1：根据名称[dataset_mode]创建一个数据集实例
        步骤2：创建一个多线程数据加载器
        """
        self.opt = opt
        dataset_class = find_dataset_using_name(opt.dataset_mode)
        self.dataset = dataset_class(opt)
        print("数据集 [%s] 已创建" % type(self.dataset).__name__)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads))

    def load_data(self):
        return self

    def __len__(self):
        """返回数据集中的数据数量"""
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        """返回一个数据批次"""
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data
