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

sys.path.remove(hairstyle_tranfer_path)

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

        # os.chdir('../hairstyle_transfer')
        print("4Current working directory: {0}".format(os.getcwd()))
        sys.path.insert(0,hairstyle_tranfer_path) # utils folder duplicating issue. 
        run_inference(mode='transfer',
                        source='./hairstyle_transfer/exp/source.png')
        # run_inference(tool=Tool(opts=None, result_path='./hairstyle_transfer/data/results/', checkpoint_path='./best_model.pt'),
        #             mode='transfer', source='./hairstyle_transfer/exp/source.png', target=rotate_path, alpha_blend=True )
        # os.chdir('../')
        print("5Current working directory: {0}".format(os.getcwd()))
        
        img1 = Image.open('./hairstyle_transfer/data/results/source_yaw_0_blend_False.jpg')
        
        ret = np.array(img1)
        return ret

    demo = gr.Interface(fn=process, inputs="text", outputs=gr.Image(type="pil"))
    demo.launch(share=False)
    # demo.launch(share=True) # uncomment this for global link