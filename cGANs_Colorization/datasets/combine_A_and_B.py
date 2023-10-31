import os
import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser('create image pairs')
parser.add_argument('--fold_A', dest='fold_A', help='input directory for image A', type=str, default='../dataset/50kshoes_edges')
parser.add_argument('--fold_B', dest='fold_B', help='input directory for image B', type=str, default='../dataset/50kshoes_jpg')
parser.add_argument('--fold_AB', dest='fold_AB', help='output directory', type=str, default='../dataset/test_AB')
args = parser.parse_args()

images = os.listdir(args.fold_A)
num_imgs = len(images)
for n in range(num_imgs):
    name_A = images[n]
    path_A = os.path.join(args.fold_A, name_A)
    path_B = os.path.join(args.fold_B, name_A)
    if os.path.isfile(path_A) and os.path.isfile(path_B):
        path_AB = os.path.join(args.fold_AB, name_A.replace('JPEG','jpg'))
        im_A = cv2.imread(path_A, 1) # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
        im_B = cv2.imread(path_B, 1) # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
        im_AB = np.concatenate([im_A, im_B], 1)
        cv2.imwrite(path_AB, im_AB)
