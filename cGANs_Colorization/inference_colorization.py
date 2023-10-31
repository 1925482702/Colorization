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

# ����һ��ArgumentParser�������ڽ��������в���
parser = argparse.ArgumentParser('inference images')

# ��������в�������������ͼ��·����ģ��·�������Ŀ¼���趨��������Ĭ��ֵ
parser.add_argument('--imagepath', dest='imagepath', help='input image file path', type=str, default='myimages/human_color/0100f95d4464c7a8012187f4c26ef8.jpg@1280w_1l_2o_100sh.jpg')
parser.add_argument('--modelpath', dest='modelpath', help='input directory for modelweights', type=str, default='checkpoints/oxford102flower_colorization/celeba_net_G.pth')
parser.add_argument('--dstroot', dest='dstroot', help='output directory', type=str, default='myresults/human_result')
args = parser.parse_args()  # ���������в��������ؽ��

# ��ȡģ��·��������ͼ��·�������Ŀ¼
model_path = args.modelpath
image_path = args.imagepath
dstdir = args.dstroot

# ����ģ��·����ȡģ��������Ϊ������Ŀ¼��
dstdir = os.path.join(args.dstroot, model_path.split('/')[-1].split('_')[0])

# ������Ŀ¼�����ڣ��򴴽�
if not os.path.exists(dstdir):
    os.mkdir(dstdir)

input_nc = 1  # ����ͨ�������Ҷ�ͼLͨ��
output_nc = 2  # ���ͨ����
ngf = 64  # ���������һ�������ͨ����
netG = 'unet_256'  # ʹ�õ���������������Ϊunet_256
norm = 'batch'  # ʹ�õĹ�һ����ʽΪbatch normalization

# ʹ��define_G��������������ģ�ͣ���ָ������ṹ�Ͳ���
modelG = define_G(input_nc, output_nc, ngf, netG, norm, True)

# ����Ԥѵ��ģ�͵Ĳ���
params = torch.load(model_path, map_location='cpu')
modelG.load_state_dict(params)

# ��ָ����ͼ���ļ�
try:
    img = Image.open(image_path)
except Exception as e:
    print(f"Error opening image: {e}")
    exit()

# ��ͼ��ת��ΪRGB��ʽ
img = img.convert('RGB')
transform = transforms.Resize((1024, 1024), Image.BICUBIC)
img = transform(img)

# ��ͼ��ת��ΪNumPy����
img = np.array(img)

# ��RGBͼ��ת��ΪLabɫ�ʿռ䣬��������������ת��
lab = color.rgb2lab(img).astype(np.float32)

# ��Labͼ��ת��ΪPyTorch��Tensor��ʽ
lab_t = transforms.ToTensor()(lab)

# ��ȡLͨ�������й�һ��
L = lab_t[[0], ...] / 50.0 - 1.0
L = L.unsqueeze(0)

# ��Lͨ�����뵽������ģ���У���ȡ���ɵ�ABͨ��
AB = modelG(L)

# ���з���һ��
AB2 = AB * 110.0
L2 = (L + 1.0) * 50.0

# ��Lͨ�������ɵ�ABͨ���ϲ�
Lab = torch.cat([L2, AB2], dim=1)
Lab = Lab[0].data.cpu().float().numpy()
Lab = np.transpose(Lab.astype(np.float64), (1, 2, 0))

# ��Lab��ʽת��ΪRGB��ʽ
rgb = (color.lab2rgb(Lab) * 255).astype(np.uint8)

# ��RGB��ʽת��ΪBGR��ʽ�����ڱ���
result = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

# ������ɫ��������Ŀ¼
output_path = os.path.join(dstdir, os.path.basename(image_path))
cv2.imwrite(output_path, result)

print(f"Colorized image saved at: {output_path}")
