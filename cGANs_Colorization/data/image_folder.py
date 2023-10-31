"""一个修改过的图像文件夹类

我们修改了官方的PyTorch图像文件夹类（https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py）
以便该类可以从当前目录及其子目录加载图像。
"""

# 重载了pytorch里的image_folder类
import torch.utils.data as data

from PIL import Image
import os
import os.path

## 支持的图像类型
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

## 返回图像路径
def make_dataset(dir, max_dataset_size=float("inf")):# 输入一个路径和最大数量（无限）
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))] # 遍历图片


def default_loader(path):
    return Image.open(path).convert('RGB')

## 重载imagefolder类
class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index] # 获得路径
        img = self.loader(path) # 加载图像
        if self.transform is not None: # 调用transform处理图像
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)