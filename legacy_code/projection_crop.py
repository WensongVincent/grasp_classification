import cv2
import os
import numpy as np
import matplotlib
from pathlib import Path
import json
import copy
from tqdm import tqdm
import shutil
import random

IMAGE_PATH = '/mnt/afs/huwensong/data/datasets/6.PieceGrasp/20240618_R3v0.3_6.PieceGrasp_P1/6.PieceGrasp/Data/NormalData/NormalLight/R3_EVT1_20240618/Pieces-Align/Occ-RoboticArm/Camera_01R2_G2704863'
RT_PATH = '/mnt/afs/zhouzhijie/data/6.PieceGrasp'
K_PATH = '/mnt/afs/share_data/R3/v0.3/0.calibration/EVT1_Cam_TopR1+R2/Camera_02TopR1_H00027'
POS_PATH = '/mnt/afs/huwensong/data/datasets/6.PieceGrasp/20240618_R3v0.3_6.PieceGrasp_P1/6.PieceGrasp/Data/NormalData/NormalLight/R3_EVT1_20240618/Pieces-Align/Occ-RoboticArm/Camera_01R2_G2704863'
OUTPUT_PATH = '/mnt/afs/huwensong/workspace/R3_grasp_classification/data_preprocessed'



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
    imagePoint = tuple(np.round(imagePoint.squeeze()).astype(int))
    return imagePoint

def crop_image(image, imagePoint, half_crop_size = 80):
    image_crop = copy.copy(image)
    image_crop = image_crop[imagePoint[1] - half_crop_size + 50 : imagePoint[1] + half_crop_size + 50, imagePoint[0] - half_crop_size : imagePoint[0] + half_crop_size]
    return image_crop

def main():

    # Read intrinsics
    intrinsics_path = Path(K_PATH)
    intrinsics = intrinsics_path.rglob(r'*.json')
    intrinsic = [intrinsic for intrinsic in intrinsics][0]

    with open(intrinsic) as f:
        intrinsic = json.load(f)
    cameraMatrix = np.array(intrinsic['mtx']).squeeze()
    distCoeffs = np.array(intrinsic['dist']).squeeze()

    # Read images
    image_path = Path(IMAGE_PATH)
    images = image_path.rglob(r'*.png')

    # Read RT
    rt_path = Path(RT_PATH)
    rts = rt_path.rglob(r'*.json')

    # Read position
    pos_path = Path(POS_PATH)
    poss = pos_path.rglob(r'*position.json')
    
    # temp = copy.copy(images)
    # num = len(list(temp))
    temp = image_path.rglob(r'*.png')
    num = len(list(temp))

    for image, rt, pos in tqdm(zip(images, rts, poss), total=num):
    # for image, rt, pos in tqdm(zip(images, rts, poss), total=num):
        # print(image) # make sure the order is correct
        # print(rt)
        piece_image = image.parts[-3]
        id_image = image.parts[-2]
        piece_rt = rt.parts[-3]
        id_rt = rt.parts[-2]
        piece_pos = pos.parts[-3]
        id_pos = pos.parts[-2]

        if (piece_image != piece_rt != piece_pos) or (id_image != id_rt != id_pos):
            print('Mismatch!')
            continue
        
        # load images, rt, and position
        image = cv2.imread(image)

        with open(rt) as f:
            rt = json.load(f)
        rvec = np.array(rt['R']).squeeze()
        tvec = np.array(rt['T']).squeeze()

        with open(pos) as f:
            pos = json.load(f)
        
        x = pos['position'][0]
        y = pos['position'][1]
        z = pos['position'][2]
        
        if (x == 2 and y == 0) or (x == 3 and y == 0) or (x == 2 and y == 1) or (x == 3 and y == 1):
            continue
        
        x, y, z = board_to_world(x, y, z)
        imagePoint = world_to_image(x, y, z, rvec, tvec, cameraMatrix, distCoeffs)

        # mark image
        image_show = copy.copy(image)
        image_show = cv2.circle(image_show, imagePoint, 2, (0, 255, 0), -1)

        # cropped image
        image_crop = crop_image(image, imagePoint, half_crop_size=80)

        
        # make directory
        os.makedirs(f'{OUTPUT_PATH}/center_mark/0', exist_ok=True)
        os.makedirs(f'{OUTPUT_PATH}/center_mark/1', exist_ok=True)
        os.makedirs(f'{OUTPUT_PATH}/gt/0', exist_ok=True)
        os.makedirs(f'{OUTPUT_PATH}/gt/1', exist_ok=True)
        os.makedirs(f'{OUTPUT_PATH}/cropped/0', exist_ok=True)
        os.makedirs(f'{OUTPUT_PATH}/cropped/1', exist_ok=True)

        
        # save croped image and gt
        if piece_image == 'None':
            cv2.imwrite(f'{OUTPUT_PATH}/center_mark/0/{piece_image}_{id_image}_center.png', image_show)
            cv2.imwrite(f'{OUTPUT_PATH}/cropped/0/{piece_image}_{id_image}_cropped.png', image_crop)
            # gt
            content = {'GT': 0}
            content = json.dumps(content, indent=4)
            with open(f'{OUTPUT_PATH}/gt/0/{piece_image}_{id_image}_cropped.json', 'w') as f:
                f.write(content)
        else:
            cv2.imwrite(f'{OUTPUT_PATH}/center_mark/1/{piece_image}_{id_image}_center.png', image_show)
            cv2.imwrite(f'{OUTPUT_PATH}/cropped/1/{piece_image}_{id_image}_cropped.png', image_crop)
            # gt
            content = {'GT': 1}
            content = json.dumps(content, indent=4)
            with open(f'{OUTPUT_PATH}/gt/1/{piece_image}_{id_image}_cropped.json', 'w') as f:
                f.write(content)


        
        # print('wait')
        


if __name__ == '__main__':
    main()