import os
from PIL import Image 
import torchvision.transforms as transforms
import cv2
import numpy as np
import torch
from models.networks import define_G, define_D
import argparse
parser = argparse.ArgumentParser('inference images')
parser.add_argument('--modelGpath', dest='modelGpath', help='input directory for modelweights', type=str, default='checkpoints/celeba_colorization/latest_net_G.pth')
parser.add_argument('--modelDpath', dest='modelDpath', help='input directory for modelweights', type=str, default='checkpoints/celeba_colorization/50_net_D.pth')
args = parser.parse_args()

modelG_path = args.modelGpath
modelD_path = args.modelDpath

input_nc = 1
output_nc = 2
ngf = 64
ndf = 64
netG = 'unet_256'
netD = 'basic' ## 70x70 PatchGAN
norm = 'batch'

modelG = define_G(input_nc, output_nc, ngf, netG, norm = norm, use_dropout = True)
modelD = define_D(input_nc + output_nc, ndf, netD, norm = norm)

paramsG = torch.load(modelG_path,map_location='cpu')
modelG.load_state_dict(paramsG)
dummy_inputG = torch.randn((1,1,256,256))
torch.onnx.export(modelG, dummy_inputG, "netG.onnx", verbose=True)

paramsD = torch.load(modelD_path,map_location='cpu')
modelD.load_state_dict(paramsD)
dummy_inputD = torch.randn((1,3,256,256))
torch.onnx.export(modelD, dummy_inputD, "netD.onnx", verbose=True)

