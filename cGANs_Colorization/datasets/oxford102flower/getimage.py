import cv2
import sys
import os
images = os.listdir(sys.argv[1])
for image in images:
    img = cv2.imread(os.path.join(sys.argv[1],image),1)
    h,w,c = img.shape
    print(img.shape)
    newimg = img[0:h,int(w/2):w,:]
    print(newimg.shape)
    cv2.imwrite(os.path.join(sys.argv[2],image),newimg)
