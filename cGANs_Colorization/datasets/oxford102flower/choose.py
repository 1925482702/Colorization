import shutil
import os
import sys
images = os.listdir(sys.argv[1])
for image in images:
    if os.path.exists(os.path.join(sys.argv[2],image)):
        shutil.move(os.path.join(sys.argv[1],image),os.path.join(sys.argv[3],image))
