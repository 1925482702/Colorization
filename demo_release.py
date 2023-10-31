# -*- coding: gbk -*-

import argparse
import matplotlib.pyplot as plt

from colorizers import *

parser = argparse.ArgumentParser()
parser.add_argument('-i','--img_path', type=str, default='imgs/tt.jpg')
parser.add_argument('--use_gpu', action='store_true', help='whether to use GPU')
opt = parser.parse_args()
opt.save_prefix = 'imgs_out/' + opt.img_path.split('/')[-1].split('.')[0]  # 修改此行以设置保存路径和文件名

# 加载模型
colorizer_cnn16 = CNN(pretrained=True).eval()
colorizer_cnn_shrot17 = CNN_Short(pretrained=True).eval()
if(opt.use_gpu):
	colorizer_cnn16.cuda()
	colorizer_cnn_shrot17.cuda()


# 默认处理图像大小为256x256
# 获取原始（"orig"）和调整大小（"rs"）分辨率下的亮度通道
img = load_img(opt.img_path)
(tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
if(opt.use_gpu):
	tens_l_rs = tens_l_rs.cuda()

# 上色器输出256x256的ab通道映射
# 调整大小并与原始L通道连接
img_bw = postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig,0*tens_l_orig),dim=1))
out_img_cnn = postprocess_tens(tens_l_orig, colorizer_cnn16(tens_l_rs).cpu())
out_img_cnn_short = postprocess_tens(tens_l_orig, colorizer_cnn_shrot17(tens_l_rs).cpu())


plt.imsave('%s_result.png' % opt.save_prefix, out_img_cnn)
plt.imsave('%s_short_result.png' % opt.save_prefix, out_img_cnn_short)

plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
plt.imshow(img)
plt.title('Original')
plt.axis('off')

plt.subplot(2,2,2)
plt.imshow(img_bw)
plt.title('Input')
plt.axis('off')

plt.subplot(2,2,3)
plt.imshow(out_img_cnn)
plt.title('Output (CNN)')
plt.axis('off')

plt.subplot(2,2,4)
plt.imshow(out_img_cnn_short)
plt.title('Output (CNN_short)')
plt.axis('off')
plt.show()
