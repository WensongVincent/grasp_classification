# Organize files into train and test metadata

import cv2
import os
import numpy as np
from pathlib import Path, PosixPath
import json
import copy
from tqdm import tqdm
import csv
from PIL import Image

from data_utils import board_to_world, world_to_image, read_jsonl_file

# RT_PATH = '/mnt/afs/huwensong/data/datasets/6.PieceGrasp_RT'
RT_PATH_LIST = ['/mnt/afs/zhouzhijie/data/temp/20240618_R3v0.3_6.PieceGrasp_TestData_P1_hhh.jsonl',
                '/mnt/afs/zhouzhijie/data/temp/20240618_R3v0.3_6.PieceGrasp_TrainData_P1_hhh.jsonl',
                '/mnt/afs/zhouzhijie/data/temp/20240703_R3v0.3_6.PieceGrasp_ClawIntersecting_TestData_P1_hhh.jsonl',
                '/mnt/afs/zhouzhijie/data/temp/20240703_R3v0.3_6.PieceGrasp_ClawIntersecting_TrainData_P1_hhh.jsonl',
                '/mnt/afs/zhouzhijie/data/temp/20240703_R3v0.3_6.PieceGrasp_ClawIntersecting_TrainData_P1_hhh.jsonl',
                '/mnt/afs/zhouzhijie/data/temp/20240703_R3v0.3_6.PieceGrasp_FreeArea_TestData_P1_hhh.jsonl',
                '/mnt/afs/zhouzhijie/data/temp/20240703_R3v0.3_6.PieceGrasp_FreeArea_TrainData_P1_hhh.jsonl',
                '/mnt/afs/zhouzhijie/data/temp/20240703_R3v0.3_6.PieceGrasp_FreeArea_TrainData_P1_hhh.jsonl',
                '/mnt/afs/zhouzhijie/data/temp/20240722_R3v0.3_6.PieceGrasp_TestData_P2_hhh.jsonl',
                '/mnt/afs/zhouzhijie/data/temp/20240722_R3v0.3_6.PieceGrasp_TrainData_P2_hhh.jsonl',
                '/mnt/afs/zhouzhijie/data/temp/20240722_R3v0.3_6.PieceGrasp_TrainData_P2_hhh.jsonl']
K_PATH = '/mnt/afs/share_data/R3/v0.3/0.calibration/EVT1_Cam_TopR1+R2/Camera_01R2_G2704863'
OUTPUT_PATH = '/mnt/afs/huwensong/workspace/R3_grasp_classification/meta_data'
OUTPUT_VERSION = '0806_1' ###### CHANGE this for new metadata ######


def main():
    test_content = []
    train_content = []

    # Read intrinsics
    intrinsics_PATH = Path(K_PATH)
    intrinsics_paths = intrinsics_PATH.rglob(r'*.json')
    intrinsics_path = [intrinsics_path for intrinsics_path in intrinsics_paths][0]

    with open(intrinsics_path) as f:
        intrinsics = json.load(f)
    cameraMatrix_ = np.array(intrinsics['mtx']).squeeze()
    distCoeffs_ = np.array(intrinsics['dist']).squeeze()

    image_list = []
    rt_dict = {}
    for rt_path in RT_PATH_LIST:
        # Read images and RT
        rt_path = PosixPath(rt_path)
        rt_dict_temp, image_list_temp = read_jsonl_file(rt_path)
        rt_dict.update(rt_dict_temp)
        image_list.append(image_list_temp)
    image_list = [PosixPath(item) for sublist in image_list for item in sublist]

    length = len(image_list)

    # Process the directories to metadata
    for image_path in tqdm(image_list, total=length):
        image_path_str = str(image_path)
        pos_path = list(Path(image_path_str).parent.rglob('*.csv'))[0]

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
            image_path_str = '/'.join(['/mnt/afs/huwensong/data/datasets/6.PieceGrasp'] + list(image_path.parts[7:]))
            rvec = np.array(rt_dict[image_path_str]['R']).squeeze()
            tvec = np.array(rt_dict[image_path_str]['T']).squeeze()

        # read mtx dist
        try:
            cameraMatrix = np.array(rt_dict[image_path_str]['mtx']).squeeze()
            distCoeffs = np.array(rt_dict[image_path_str]['dist']).squeeze()
        except:
            cameraMatrix = cameraMatrix_
            distCoeffs = distCoeffs_

        # Calculate anchor

        ###### Mind this ######
        # read position
        # x, y, z_in_mm = (None, None, 58.0) 
        # x, y, z_in_mm = (None, None, 68.0) 
        x, y, z_in_mm = (None, None, 78.0) 
        ###### Mind this ######


        with open(pos_path, 'r') as f:
            reader = csv.reader(f)
            nums = list(reader)
            try:
                x, y = [float(num) for num in nums[0]]
            except:
                x, y, _, _, _ = [float(num) for num in nums[0]]

        # exclude the position that cannot see the gripper
        if (x == 2.0 and y == 0.0) or (x == 3.0 and y == 0.0) or (x == 2.0 and y == 1.0) or (x == 3.0 and y == 1.0) or (x == 1.0 and y == 0.0):
            continue
        
        # transformation
        x_in_mm, y_in_mm, z_in_mm = board_to_world(x, y, z_in_mm)
        anchor = list(world_to_image(x_in_mm, y_in_mm, z_in_mm, rvec, tvec, cameraMatrix, distCoeffs))

        ###### Mind this ######
        # box = [160, 50, 0] # [crop_size, v_trans, h_trans], see crop_image for meaning 
        # box = [224, 0, 0] 
        # box = [190, 0, 0] 
        box = [160, 0, 0] 
        # box = [120, 0, 0] 
        ###### Mind this ######

        # Get gt
        gt = None
        if 'Ungrabed' in image_path.parts:
            gt = 0
        elif 'None' in image_path.parts:
            gt = 0
        else:
            gt = 1
            # raise('Wrong data format')


        # Save metadata based on train or test
        content = {'imagePath': image_path_str,
                'gt': gt,
                'anchor': anchor,
                'box': box}

        if 'TestData' in image_path.parts:
            test_content.append(content)
        elif 'TrainData' in image_path.parts:
            train_content.append(content)


    # Write into file
    with open(f'{OUTPUT_PATH}/metadata_{OUTPUT_VERSION}_test.json', 'w') as f:
        json.dump(test_content, f, ensure_ascii=False, indent=4)
    with open(f'{OUTPUT_PATH}/metadata_{OUTPUT_VERSION}_train.json', 'w') as f:
        json.dump(train_content, f, ensure_ascii=False, indent=4)
    
    '''
    saved format:
    [
    {
        "imagePath": "path/to/image",
        "gt": 1,
        "anchor": [
            1273,
            220
        ],
        "box": [
            96,
            0,
            0
        ]
    },
    ...
    ]

    '''


if __name__ == '__main__' :
    main()