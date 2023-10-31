import cv2
import sys
import numpy as np
img1 = cv2.imread(sys.argv[1],1)
img2 = cv2.imread(sys.argv[2],1)
diff = abs(img1 - img2).astype(np.uint8)
result = np.concatenate([img1, img2, diff], 1)
print(np.max(diff))
cv2.imshow("diff",result)
cv2.waitKey(0)
