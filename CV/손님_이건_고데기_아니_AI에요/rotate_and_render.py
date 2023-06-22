ddfa_path = '/home/aiku/AIKU/hair/rotateandrender/3ddfa'
module_path ='/home/aiku/AIKU/hair/rotateandrender/'
import sys
import os
sys.path.insert(0,ddfa_path)
sys.path.insert(0,module_path)

# print(sys.path)
from data import curve
from rotateandrender.rotate_and_render import inference_3d
from inference import run_3ddfa


if __name__ == '__main__':
    # Import the os module
    import os

    # Print the current working directory
    # print("Current working directory: {0}".format(os.getcwd()))

    # Change the current working directory
    os.chdir('./rotateandrender/3ddfa')

    run_3ddfa()


    # sys.path.append(module_path)
    # Change the current working directory
    os.chdir('../')
    # print("Current working directory: {0}".format(os.getcwd()))


    inference_3d()