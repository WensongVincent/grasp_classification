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

IMAGE_PATH =  '/mnt/afs/huwensong/data/datasets/6.PieceGrasp'
POS_PATH = '/mnt/afs/huwensong/data/datasets/6.PieceGrasp'
RT_PATH = '/mnt/afs/huwensong/data/datasets/6.PieceGrasp_RT'
K_PATH = '/mnt/afs/share_data/R3/v0.3/0.calibration/EVT1_Cam_TopR1+R2/Camera_02TopR1_H00027'
OUTPUT_PATH = '/mnt/afs/huwensong/workspace/R3_grasp_classification/meta_data'
OUTPUT_VERSION = '0710'


def board_to_world(x, y, z_in_mm): 
    '''
    Project poses from board (unit: block) frame to world frame (unit: mm)
    Requires: board block size in mm
    Input: x, y in block in board frame, z in mm
    Output: x, y, z in mm in world frame 
    '''
    # board setting
    board_edge = 0.731
    board_block = 35
    board_line = 0.5
    num_block = 7

    new_board_edge = board_edge - board_line
    new_board_block = board_block + board_line

    # conversion
    mm_to_block = 1 / new_board_block
    block_to_mm = new_board_block

    # mapping
    x_temp = -(x - num_block) 
    y_temp = y
    x_temp = x_temp + (new_board_block / 2) * mm_to_block + (new_board_edge / 2) * mm_to_block
    y_temp = y_temp + (new_board_block / 2) * mm_to_block + (new_board_edge / 2) * mm_to_block

    x_in_mm = x_temp * block_to_mm
    y_in_mm = y_temp * block_to_mm
    return (x_in_mm, y_in_mm, - z_in_mm)



def world_to_image(x_in_mm, y_in_mm, z_in_mm, rvec, tvec, cameraMatrix, distCoeffs):
    '''
    Project position from world frame to images
    Requires: R, T, MTX, DIST
    Input: x, y, z in mm in world frame
    Output: u, v in pixel frame
    '''
    imagePoint, _ = cv2.projectPoints(np.array([x_in_mm, y_in_mm, z_in_mm]), rvec, tvec, cameraMatrix, distCoeffs)
    imagePoint = list(np.round(imagePoint.squeeze()))
    return [int(point) for point in imagePoint]


def read_jsonl_file(jsonl_file_path):
    data_dict = {}
    image_list = []
    try:
        with open(jsonl_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    data = json.loads(line)
                    imgpath = data.get('imgpath')
                    h_info = data.get('h_info', {})
                    if imgpath:
                        data_dict[imgpath] = h_info
                        image_list.append(imgpath)
                except json.JSONDecodeError:
                    print('Invalid JSON line')
    except FileNotFoundError:
        print(f'File not found: {jsonl_file_path}')
    return data_dict, image_list

def crop_image(image, imagePoint, crop_size = 160, v_trans = 50, h_trans = 0):
    '''
    crop_size: image size after crop
    v_trans: downward (larger number) translation of the crop box
    h_trans: rightward (larger number) translation of the crop box
    '''
    image_crop = copy.copy(image)
    crop_size /= 2 # this is actually half_crop_size
    image_crop = image_crop[imagePoint[1] - crop_size + v_trans : imagePoint[1] + crop_size + v_trans, imagePoint[0] - crop_size + h_trans : imagePoint[0] + crop_size + h_trans]
    return image_crop

def crop_image_PIL(image: Image, imagePoint, box = [160, 50, 0]):
    '''
    imagePoint: anchor [x, y]
    box: [
            crop_size: image size after crop
            v_trans: downward (larger number) translation of the crop box
            h_trans: rightward (larger number) translation of the crop box
            ]
    '''
    crop_size, v_trans, h_trans = box
    crop_size /= 2 # this is actually half_crop_size
    y1 = imagePoint[1] - crop_size + v_trans
    y2 = imagePoint[1] + crop_size + v_trans
    x1 = imagePoint[0] - crop_size + h_trans
    x2 = imagePoint[0] + crop_size + h_trans
    image_crop = image.crop((x1, y1, x2, y2))
    return image_crop


def main():
    test_content = []
    train_content = []

    # Read intrinsics
    intrinsics_PATH = Path(K_PATH)
    intrinsics_paths = intrinsics_PATH.rglob(r'*.json')
    intrinsics_path = [intrinsics_path for intrinsics_path in intrinsics_paths][0]

    with open(intrinsics_path) as f:
        intrinsics = json.load(f)
    cameraMatrix = np.array(intrinsics['mtx']).squeeze()
    distCoeffs = np.array(intrinsics['dist']).squeeze()

    # # Read images
    # image_PATH = Path(IMAGE_PATH)
    # image_paths = image_PATH.rglob(r'*.png')

    # Read images and RT
    image_list = []
    rt_dict = {}
    rt_PATH = Path(RT_PATH)
    rt_paths = rt_PATH.rglob(r'*.jsonl')
    for rt_path in rt_paths:
        rt_dict_temp, image_list_temp = read_jsonl_file(rt_path)
        rt_dict.update(rt_dict_temp)
        image_list.append(image_list_temp)
    image_list = [PosixPath(item) for sublist in image_list for item in sublist]


    # # Read position
    # pos_PATH = Path(POS_PATH)
    # pos_paths = pos_PATH.rglob(r'*.csv')
    
    # temp = copy.copy(images)
    # num = len(list(temp))
    # temp = image_PATH.rglob(r'*.png')
    # num = len(list(temp))
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
        
        # transformation
        x_in_mm, y_in_mm, z_in_mm = board_to_world(x, y, z_in_mm)
        anchor = list(world_to_image(x_in_mm, y_in_mm, z_in_mm, rvec, tvec, cameraMatrix, distCoeffs))
        box = [160, 50, 0] # [crop_size, v_trans, h_trans], see crop_image for meaning

        # Get gt
        gt = None
        if 'Grabbed' in image_path.parts:
            gt = 1
        elif 'Ungrabed' in image_path.parts:
            gt = 0
        else:
            raise('Wrong data format')


        # Save metadata based on train or test
        content = {'imagePath': image_path_str,
                    'gt': gt,
                    'anchor': anchor,
                    'box': box}

        if 'TestData' in image_path.parts:
            test_content.append(content)
        elif 'TrainData' in image_path.parts:
            train_content.append(content)

    with open(f'{OUTPUT_PATH}/metadata_{OUTPUT_VERSION}_test.json', 'w') as f:
        json.dump(test_content, f, ensure_ascii=False, indent=4)
    with open(f'{OUTPUT_PATH}/metadata_{OUTPUT_VERSION}_train.json', 'w') as f:
        json.dump(train_content, f, ensure_ascii=False, indent=4)
    
    '''
    saved format:
    [
    {
        "imagePath": "/mnt/afs/huwensong/data/datasets/6.PieceGrasp/20240618_R3v0.3_6.PieceGrasp_TrainData_P1/6.PieceGrasp/TrainData/NormalData/NormalLight/R3_EVT1_20240618/Pieces-Align/Occ-RoboticArm/Camera_01R2_G2704863/Grabbed/B-Bishop/1718691791919/image2.png",
        "gt": 1,
        "anchor": [
            1273,
            220
        ],
        "box": [
            160,
            50,
            0
        ]
    },
    ...
    ]

    '''


if __name__ == '__main__' :
    main()