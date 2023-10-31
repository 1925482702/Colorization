# -*- coding: gbk -*-

import os
from PIL import Image
import torchvision.transforms as transforms
import cv2
import numpy as np
import torch
from skimage import color
from cGANs_Colorization.models.networks import define_G
import argparse

# 创建一个ArgumentParser对象，用于解析命令行参数
parser = argparse.ArgumentParser('inference images')

# 添加命令行参数，包括输入图像路径、模型路径、输出目录的设定，并设置默认值
parser.add_argument('--imagepath', dest='imagepath', help='input image file path', type=str, default='myimages/human_color/0100f95d4464c7a8012187f4c26ef8.jpg@1280w_1l_2o_100sh.jpg')
parser.add_argument('--modelpath', dest='modelpath', help='input directory for modelweights', type=str, default='checkpoints/oxford102flower_colorization/celeba_net_G.pth')
parser.add_argument('--dstroot', dest='dstroot', help='output directory', type=str, default='myresults/human_result')
args = parser.parse_args()  # 解析命令行参数并返回结果

# 获取模型路径、输入图像路径、输出目录
model_path = args.modelpath
image_path = args.imagepath
dstdir = args.dstroot

# 根据模型路径提取模型名称作为结果输出目录名
dstdir = os.path.join(args.dstroot, model_path.split('/')[-1].split('_')[0])

# 如果输出目录不存在，则创建
if not os.path.exists(dstdir):
    os.mkdir(dstdir)

input_nc = 1  # 输入通道数，灰度图L通道
output_nc = 2  # 输出通道数
ngf = 64  # 生成器最后一个卷积层通道数
netG = 'unet_256'  # 使用的生成器网络类型为unet_256
norm = 'batch'  # 使用的归一化方式为batch normalization

# 使用define_G函数构建生成器模型，并指定网络结构和参数
modelG = define_G(input_nc, output_nc, ngf, netG, norm, True)

# 加载预训练模型的参数
params = torch.load(model_path, map_location='cpu')
modelG.load_state_dict(params)

# 打开指定的图像文件
try:
    img = Image.open(image_path)
except Exception as e:
    print(f"Error opening image: {e}")
    exit()

# 将图像转换为RGB格式
img = img.convert('RGB')
transform = transforms.Resize((1024, 1024), Image.BICUBIC)
img = transform(img)

# 将图像转换为NumPy数组
img = np.array(img)

# 将RGB图像转换为Lab色彩空间，并进行数据类型转换
lab = color.rgb2lab(img).astype(np.float32)

# 将Lab图像转换为PyTorch的Tensor格式
lab_t = transforms.ToTensor()(lab)

# 提取L通道并进行归一化
L = lab_t[[0], ...] / 50.0 - 1.0
L = L.unsqueeze(0)

# 将L通道输入到生成器模型中，获取生成的AB通道
AB = modelG(L)

# 进行反归一化
AB2 = AB * 110.0
L2 = (L + 1.0) * 50.0

# 将L通道与生成的AB通道合并
Lab = torch.cat([L2, AB2], dim=1)
Lab = Lab[0].data.cpu().float().numpy()
Lab = np.transpose(Lab.astype(np.float64), (1, 2, 0))

# 将Lab格式转换为RGB格式
rgb = (color.lab2rgb(Lab) * 255).astype(np.uint8)

# 将RGB格式转换为BGR格式，用于保存
result = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

# 保存上色结果到输出目录
output_path = os.path.join(dstdir, os.path.basename(image_path))
cv2.imwrite(output_path, result)

print(f"Colorized image saved at: {output_path}")
