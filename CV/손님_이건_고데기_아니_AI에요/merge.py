'''
import gradio as gr

import numpy as np
from PIL import Image
import torch
import shutil

hairstyle_tranfer_path = '/home/aiku/AIKU/hair/hairstyle_transfer'

# from rotateandrender.rotate_and_render import inference_3d
# module_path = '/home/aiku/AIKU/hair/rotateandrender/3ddfa'
# path2 = '/home/aiku/AIKU/hair/hairstyle_transfer'
ddfa_path = '/home/aiku/AIKU/hair/rotateandrender/3ddfa'
module_path ='/home/aiku/AIKU/hair/rotateandrender/'
import sys
import os
sys.path.insert(0,ddfa_path)
sys.path.insert(0,module_path)

from data import curve
from rotateandrender.rotate_and_render import inference_3d
from inference import run_3ddfa
sys.path.insert(2,hairstyle_tranfer_path)


#!
if os.path.exists('exp'):
    shutil.rmtree('exp')
    os.mkdir('exp')

if os.path.exists('data'):
    shutil.rmtree('data')
# 
# source image(사용자 이미지)
source_path = '/home/aiku/AIKU/hair/source3.png'
img1 = Image.open(source_path)
# img1 = img1.resize((1024, 1024))
img1.save('/home/aiku/AIKU/hair/hairstyle_transfer/exp/source.png')
#
# target image(헤어스타일 이미지)
target_path = '/home/aiku/AIKU/hair/target8.png'
img2 = Image.open(target_path)
# img2 = img2.resize((1024, 1024))
img2.save('/home/aiku/AIKU/hair/hairstyle_transfer/exp/target.png')
#
from argparse import ArgumentParser
from unittest import result

from hairstyle_transfer.hairstyle_transfer_tool import Tool

def run_inference(tool, mode, source, target, alpha_blend, n_steps_interp = None, face_interp = None):
    if mode == 'transfer':
        paths_to_results = tool.hairstyle_transfer(source, target, alpha_blend = alpha_blend)
    elif mode == 'interp':
        paths_to_results = tool.interpolation_single_pair(source, target, n_steps=n_steps_interp, interpolate_hair = not face_interp, alpha_blend = alpha_blend)
    elif mode == 'manip':
        paths_to_results = tool.hair_manipulation_single(source, args['manip_attribute'], coeffs=args['manip_strength'], alpha_blend= alpha_blend)
    else:
        raise 'Mode not recognized'
    return paths_to_results
#!

process1_res_path = '/home/aiku/AIKU/hair/rotateandrender/results/rs_model/example/orig/yaw_0.0_input.png'
rotate_path = '/home/aiku/AIKU/hair/rotateandrender/results/rs_model/example/aligned/yaw_0.0_input.png'
result_path = './data/results/'

def process(input):
    # print("Current working directory: {0}".format(os.getcwd()))
    os.chdir('./rotateandrender/3ddfa')
    run_3ddfa()

    os.chdir('../')
    inference_3d()
    
    # run_inference(tool=Tool(opts=None, result_path='./data/results/', checkpoint_path='./best_model.pt'),
    #            mode='transfer', source='./exp/target.png', target='./exp/source.png', alpha_blend=True )
    run_inference(tool=Tool(opts=None, result_path='./hairstyle_transfer/data/results/', checkpoint_path='./best_model.pt'),
               mode='transfer', source='./exp/target.png', target='./exp/source.png', alpha_blend=True )
    print("Current working directory: {0}".format(os.getcwd()))
    img1 = Image.open('./hairstyle_transfer/data/results/target_source_blend_False.jpg')
    return img1

print("1Current working directory: {0}".format(os.getcwd()))
os.chdir('./rotateandrender/3ddfa')
# os.chdir('./3ddfa')
run_3ddfa()
print("2Current working directory: {0}".format(os.getcwd()))
os.chdir('../')

print("3Current working directory: {0}".format(os.getcwd()))
inference_3d()
os.chdir('../')
print("4Current working directory: {0}".format(os.getcwd()))

# run_inference(tool=Tool(opts=None, result_path='./data/results/', checkpoint_path='./best_model.pt'),
#            mode='transfer', source='./exp/target.png', target='./exp/source.png', alpha_blend=True )
# run_inference(tool=Tool(opts=None, result_path='./hairstyle_transfer/data/results/', checkpoint_path='./best_model.pt'),
#             mode='transfer', source='./hairstyle_transfer/exp/source.png', target=rotate_path, alpha_blend=True )
print("5Current working directory: {0}".format(os.getcwd()))
img1 = Image.open('./hairstyle_transfer/data/results/target_source_blend_False.jpg')

# demo = gr.Interface(fn=process, inputs="text", outputs="image")

# demo.launch(share=True)
'''
# * [module] rotate and render part
ddfa_path = '/home/aiku/AIKU/hair/rotateandrender/3ddfa'
module_path ='/home/aiku/AIKU/hair/rotateandrender/'
hairstyle_tranfer_path = '/home/aiku/AIKU/hair/hairstyle_transfer'
import sys
import os
sys.path.insert(0,ddfa_path)
sys.path.insert(0,module_path)
sys.path.insert(2,hairstyle_tranfer_path)
# print(sys.path)
from data import curve
from rotateandrender.rotate_and_render import inference_3d
from inference import run_3ddfa

# * [module] hair transfer part
import shutil
import torch
from PIL import Image

# * [module] others
import gradio as gr
import numpy as np
import cv2
from time import sleep
import matplotlib.pyplot as plt
# * import ends ://


# if os.path.exists('exp'):
#     shutil.rmtree('exp')
#     os.mkdir('exp')

# if os.path.exists('data'):
#     shutil.rmtree('data')
    
# from argparse import ArgumentParser
# from unittest import result

# from hairstyle_transfer.hairstyle_transfer_tool import Tool

# def run_inference(tool, mode, source, target, alpha_blend, n_steps_interp = None, face_interp = None):
#     if mode == 'transfer':
#         paths_to_results = tool.hairstyle_transfer(source, target, alpha_blend = alpha_blend)
#     elif mode == 'interp':
#         paths_to_results = tool.interpolation_single_pair(source, target, n_steps=n_steps_interp, interpolate_hair = not face_interp, alpha_blend = alpha_blend)
#     elif mode == 'manip':
#         paths_to_results = tool.hair_manipulation_single(source, args['manip_attribute'], coeffs=args['manip_strength'], alpha_blend= alpha_blend)
#     else:
#         raise 'Mode not recognized'
#     return paths_to_results

sys.path.remove(hairstyle_tranfer_path)


# if __name__ == '__main__':
# def process(input):
if __name__ == '__main__':
    def process(input):
        rotate_path = '/home/aiku/AIKU/hair/rotateandrender/results/rs_model/example/aligned/yaw_0.0_input.png'
        # Import the os module
        import os
        from hairstyle_transfer.run import run_inference
        
        
        # Print the current working directory
        # print("Current working directory: {0}".format(os.getcwd()))

        # Change the current working directory
        print("1Current working directory: {0}".format(os.getcwd()))
        os.chdir('/home/aiku/AIKU/hair/rotateandrender/3ddfa')

        run_3ddfa()
        print("2Current working directory: {0}".format(os.getcwd()))


        # sys.path.append(module_path)
        # Change the current working directory
        os.chdir('/home/aiku/AIKU/hair/rotateandrender')
        # print("Current working directory: {0}".format(os.getcwd()))

        print("3Current working directory: {0}".format(os.getcwd()))
        inference_3d()
        os.chdir('/home/aiku/AIKU/hair/')
        print('gotosleep@@@')
        sleep(1)
        print('backHome@@@')
        # os.chdir('../hairstyle_transfer')
        print("4Current working directory: {0}".format(os.getcwd()))
        sys.path.insert(0,hairstyle_tranfer_path) # utils folder duplicating issue. 
        run_inference(mode='transfer',
                        source='./hairstyle_transfer/exp/source.png')
        # run_inference(tool=Tool(opts=None, result_path='./hairstyle_transfer/data/results/', checkpoint_path='./best_model.pt'),
        #             mode='transfer', source='./hairstyle_transfer/exp/source.png', target=rotate_path, alpha_blend=True )
        # os.chdir('../')
        print("5Current working directory: {0}".format(os.getcwd()))
        # print('gotosleep@@@')
        # sleep(120)
        # print('backHome@@@')
        img1 = Image.open('./hairstyle_transfer/data/results/source_yaw_0_blend_False.jpg')
        
        ret = np.array(img1)
        print(type(img1), img1)
        print(type(ret))
        # plt.imshow(img1)
        return ret

    demo = gr.Interface(fn=process, inputs="text", outputs=gr.Image(type="pil"))
    demo.launch(share=False)
# demo.launch(share=True)