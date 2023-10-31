# import torch
# from cGANs_Colorization.models.networks import define_G
# import torchvision.transforms as transforms
# from skimage import color
# from PIL import Image
# import numpy as np
#
# # 设置模型参数
# input_nc = 1  # 输入通道数，灰度图L通道
# output_nc = 2  # 输出通道数
# ngf = 64  # 生成器最后一个卷积层通道数
# netG = 'unet_256'  # 使用的生成器网络类型为unet_256
# norm = 'batch'  # 使用的归一化方式为batch normalization
#
# # 创建和加载模型
# modelG = define_G(input_nc, output_nc, ngf, netG, norm, False)
# model_path = r'C:\Users\彭英麒\Desktop\Colorization_2\cGANs_Colorization\checkpoints\oxford102flower_colorization\building_net_G.pth'  # 更改为你的模型路径
# params = torch.load(model_path, map_location='cpu')
# modelG.load_state_dict(params)
# modelG.eval()
#
# def colorize_image(img_path):
#     # 加载选定的图像
#     img = Image.open(img_path)
#     img = img.convert('RGB')
#     img = img.resize((256, 256), Image.BICUBIC)
#     img = np.array(img)
#
#     # 转换为 Lab 色彩空间，并提取 L 通道
#     img_lab = color.rgb2lab(img).astype(np.float32)
#     img_l = img_lab[:, :, 0]
#     img_l_transformed = (transforms.ToTensor()(img_l) / 50.0) - 1.0
#     img_l_transformed = img_l_transformed.unsqueeze(0)
#
#     # 使用模型生成彩色图像
#     with torch.no_grad():
#         ab_output = modelG(img_l_transformed)
#     ab_output = (ab_output.squeeze(0).numpy().transpose(1, 2, 0) * 110.0).astype(np.float64)
#
#     # 合并L通道和生成的AB通道
#     colorized_img_lab = np.stack([img_l, ab_output[:, :, 0], ab_output[:, :, 1]], axis=-1)
#     colorized_img_rgb = (color.lab2rgb(colorized_img_lab) * 255).astype(np.uint8)
#
#     return colorized_img_rgb
