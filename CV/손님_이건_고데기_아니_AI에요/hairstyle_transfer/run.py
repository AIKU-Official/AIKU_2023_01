import os
import shutil
import torch
from PIL import Image


ddfa_path = '/home/aiku/AIKU/hair/rotateandrender/3ddfa'
module_path ='/home/aiku/AIKU/hair/rotateandrender/'
hairstyle_tranfer_path = '/home/aiku/AIKU/hair/hairstyle_transfer'
import sys
sys.path.insert(0,ddfa_path)
sys.path.insert(0,module_path)
sys.path.insert(2,hairstyle_tranfer_path)


if os.path.exists('exp'):
    shutil.rmtree('exp')
    os.mkdir('exp')

if os.path.exists('data'):
    shutil.rmtree('data')
    
    
# source image(사용자 이미지)
source_path = '/home/aiku/AIKU/hair/source6.jpg'
img1 = Image.open(source_path)
img1 = img1.resize((1024, 1024))
# img1.save('exp/source.png')

# target image(헤어스타일 이미지)
target_path = '/home/aiku/AIKU/hair/target8.png'
img2 = Image.open(target_path)
# img2 = img2.resize((1024, 1024))
# img2.save('exp/target.png')

from argparse import ArgumentParser
from unittest import result

from hairstyle_transfer_tool import Tool
rotate_path = '/home/aiku/AIKU/hair/rotateandrender/results/rs_model/example/aligned/yaw_0.0_input.png'
def run_inference(tool=Tool(
                    opts=None, 
                    result_path='/home/aiku/AIKU/hair/hairstyle_transfer/data/results', 
                    checkpoint_path='/home/aiku/AIKU/hair/hairstyle_transfer/best_model.pt'
                ),
                mode='transfer',
                source='./exp/source.png', 
                target=rotate_path, 
                alpha_blend=True, 
                n_steps_interp = None, 
                face_interp = None
    ):
    if mode == 'transfer':
        paths_to_results = tool.hairstyle_transfer(source, target, alpha_blend = alpha_blend)
    elif mode == 'interp':
        paths_to_results = tool.interpolation_single_pair(source, target, n_steps=n_steps_interp, interpolate_hair = not face_interp, alpha_blend = alpha_blend)
    elif mode == 'manip':
        paths_to_results = tool.hair_manipulation_single(source, args['manip_attribute'], coeffs=args['manip_strength'], alpha_blend= alpha_blend)
    else:
        raise 'Mode not recognized'
    return paths_to_results


# if __name__ == '__main__':
#     # args = parse_args()
#     print(f'All done')
#     tool = Tool(opts=None, result_path='./data/results/', checkpoint_path='./best_model.pt')
#     paths_to_results =run_inference(tool, mode='transfer',source='./exp/target.png', target='./exp/source.png', alpha_blend=True )
#     print(f'Results saved as: {paths_to_results}')

# run_inference(tool=Tool(opts=None, result_path='./data/results/', checkpoint_path='./best_model.pt'),
#                mode='transfer', source='./exp/target.png', target='./exp/source.png', alpha_blend=True )

# run_inference(tool=Tool(opts=None, result_path='./data/results/', checkpoint_path='./best_model.pt'),
#             mode='transfer', source='./exp/source.png', target=rotate_path, alpha_blend=True )
