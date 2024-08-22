import cv2
import os
import numpy as np
from pathlib import Path, PosixPath
import json
import copy
from tqdm import tqdm
import csv
from PIL import Image

from data_utils import read_jsonl_file
from Pipeline import Pipeline



########### copy from data_preprocess.py ########### 
IMAGE_PATH =  '/mnt/afs/huwensong/data/datasets/6.PieceGrasp'
POS_PATH = '/mnt/afs/huwensong/data/datasets/6.PieceGrasp'
RT_PATH = '/mnt/afs/huwensong/data/datasets/6.PieceGrasp_RT'
K_PATH = '/mnt/afs/share_data/R3/v0.3/0.calibration/EVT1_Cam_TopR1+R2/Camera_02TopR1_H00027'

# Read intrinsics
intrinsics_PATH = Path(K_PATH)
intrinsics_paths = intrinsics_PATH.rglob(r'*.json')
intrinsics_path = [intrinsics_path for intrinsics_path in intrinsics_paths][0]

with open(intrinsics_path) as f:
    intrinsics = json.load(f)
cameraMatrix = np.array(intrinsics['mtx']).squeeze()
distCoeffs = np.array(intrinsics['dist']).squeeze()

# Read images and RT
rt_PATH = Path(RT_PATH)
rt_paths = rt_PATH.rglob(r'*.jsonl')
image_list = []
rt_dict = {}
for rt_path in rt_paths:
    rt_dict_temp, image_list_temp = read_jsonl_file(rt_path)
    rt_dict.update(rt_dict_temp)
    image_list.append(image_list_temp)
image_list = [PosixPath(item) for sublist in image_list for item in sublist]

length = len(image_list)

# Process the directories to metadata
for image_path in tqdm(image_list, total=length):
    image_path_str = str(image_path)
    pos_path = list(Path(image_path_str).parent.rglob('*.csv'))[0]
    pos_path_str = str(pos_path)


    # exclude calibration data
    if 'calib-R2' in image_path.parts:
        continue
    
    # make sure they are corresponding
    if image_path.parts[-2] != pos_path.parts[-2]:
        print('Mismatch!')
        continue
    
    # read rt
    try:
        rvec = np.array(rt_dict[image_path_str]['R']).squeeze()
        tvec = np.array(rt_dict[image_path_str]['T']).squeeze()
    except:
        image_path_str = '/'.join([IMAGE_PATH] + list(image_path.parts[7:]))
        rvec = np.array(rt_dict[image_path_str]['R']).squeeze()
        tvec = np.array(rt_dict[image_path_str]['T']).squeeze()

    # Calculate anchor
    # read position
    x, y, z_in_mm = (None, None, 58.0)
    with open(pos_path, 'r') as f:
        reader = csv.reader(f)
        nums = list(reader)
        try:
            x, y = [float(num) for num in nums[0]]
        except:
            x, y, _, _, _ = [float(num) for num in nums[0]]

    # exclude the position that cannot see the gripper
    if (x == 2.0 and y == 0.0) or (x == 3.0 and y == 0.0) or (x == 2.0 and y == 1.0) or (x == 3.0 and y == 1.0):
        continue

########### end of copy from data_preprocess.py ########### 


    camera = {'R': rvec,
              'T': tvec,
              'cameraMatrix': cameraMatrix,
              'distCoeffs': distCoeffs}
    position = [x, y]
    image = Image.open(image_path_str).convert('RGB')

    ckpt_path = '/mnt/afs/huwensong/workspace/R3_grasp_classification/result/0710_1/ckpt.pth'
    pipline = Pipeline(ckpt_path, camera)
    cls = pipline(image, position)
    print(cls)


        