import glob
import os
basedir = os.path.dirname(__file__)
path = os.path.join(basedir, 'file_list.txt')

images = os.path.join(basedir, 'Images')
file = glob.glob(images + '/*')

with open(path, mode='w') as f:
    for fn in file :
        f.write(os.path.basename(fn) + '\n')