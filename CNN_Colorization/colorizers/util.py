
from PIL import Image
import numpy as np
from skimage import color
import torch
import torch.nn.functional as F
from IPython import embed

def load_img(img_path):# 加载图像
	out_np = np.asarray(Image.open(img_path))
	if(out_np.ndim==2):
		out_np = np.tile(out_np[:,:,None],3)
	return out_np

def resize_img(img, HW=(256,256), resample=3):  # 裁剪图像
	return np.asarray(Image.fromarray(img).resize((HW[1],HW[0]), resample=resample))

def preprocess_img(img_rgb_orig, HW=(256,256), resample=3): # 预处理图像
	# return original size L and resized L as torch Tensors
	img_rgb_rs = resize_img(img_rgb_orig, HW=HW, resample=resample)
	# 这两行将原始图像和调整大小后的图像分别转换为 LAB 色彩空间
	img_lab_orig = color.rgb2lab(img_rgb_orig)
	img_lab_rs = color.rgb2lab(img_rgb_rs)
	# 这两行提取了 LAB 色彩空间中的亮度通道，包括原始的和调整后的
	img_l_orig = img_lab_orig[:,:,0]
	img_l_rs = img_lab_rs[:,:,0]

	# 将亮度通道转换为 PyTorch 的 Tensor 对象，并调整了其维度以匹配模型的输入要求
	tens_orig_l = torch.Tensor(img_l_orig)[None,None,:,:]
	tens_rs_l = torch.Tensor(img_l_rs)[None,None,:,:]
	# 数返回了两个 Tensor 对象，分别是原始亮度通道和调整大小后的亮度通道
	return (tens_orig_l, tens_rs_l)


# tens_orig_l：原始尺寸的亮度通道，是一个形状为 1 x 1 x H_orig x W_orig 的张量。
# out_ab：模型输出的色度通道，是一个形状为 1 x 2 x H x W 的张量。
# mode：插值模式，用于调整色度通道的大小，默认为 'bilinear'。
def postprocess_tens(tens_orig_l, out_ab, mode='bilinear'):  #后处理图像
	# tens_orig_l 	1 x 1 x H_orig x W_orig
	# out_ab 		1 x 2 x H x W

	# 获取了原始亮度通道和模型输出的色度通道的空间尺寸
	HW_orig = tens_orig_l.shape[2:]
	HW = out_ab.shape[2:]

	# 判断原始亮度通道和模型输出的色度通道的空间尺寸是否相同
	# 如果尺寸不同，就调用了 PyTorch 的插值函数 F.interpolate 来将模型输出的色度通道调整为原始尺寸
	if(HW_orig[0]!=HW[0] or HW_orig[1]!=HW[1]):
		out_ab_orig = F.interpolate(out_ab, size=HW_orig, mode='bilinear')
	else:
		out_ab_orig = out_ab
	# 将调整后的色度通道与原始尺寸的亮度通道拼接在一起，形成完整的 LAB 图像
	out_lab_orig = torch.cat((tens_orig_l, out_ab_orig), dim=1)
	# 将 LAB 图像转换为 RGB 图像，并将结果返回
	return color.lab2rgb(out_lab_orig.data.cpu().numpy()[0,...].transpose((1,2,0)))
