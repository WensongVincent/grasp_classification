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
OUTPUT_PATH = '/mnt/afs/huwensong/workspace/R3_grasp_classification/meta_data'
OUTPUT_VERSION = '0814_1' ###### CHANGE this for new metadata ######


def main():
    test_content = []
    train_content = []

    image_list = []
    for rt_path in RT_PATH_LIST:
        # Read images and RT
        rt_path = PosixPath(rt_path)
        _, image_list_temp = read_jsonl_file(rt_path)
        image_list.append(image_list_temp)
    image_list = [PosixPath(item) for sublist in image_list for item in sublist]

    length = len(image_list)

    # Process the directories to metadata
    for image_path in tqdm(image_list, total=length):
        image_path_str = str(image_path)

        # exclude calibration data
        if 'calib-R2' in image_path.parts:
            continue
        
        # get label path
        label_path_str = image_path_str.replace("/media/raw_data", "/annotations/label")
        label_path_str = label_path_str + ".json"
        if not os.path.exists(label_path_str):
            continue

        with open(label_path_str, 'r') as f:
            label = json.load(f)

            # exclude unlabelled
            if label['step_1']['result'] == []:
                continue
            
            x_tl = label['step_1']['result'][0]['x']
            y_tl = label['step_1']['result'][0]['y']
            width = label['step_1']['result'][0]['width']
            height = label['step_1']['result'][0]['height']
            attribute = int(label['step_1']['result'][0]['attribute'])

        anchor = [round(x_tl + width/2), round(y_tl + height/2)]
        box = [width, 0, 0] 


        # Get gt
        gt = None
        if 'Ungrabed' in image_path.parts:
            gt = 0
        elif 'None' in image_path.parts:
            gt = 0
        else:
            gt = 1
            # raise('Wrong data format')
        
        if gt != attribute:
            print("GT and attribute don't match!")

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