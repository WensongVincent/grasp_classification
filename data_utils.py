import cv2
import os
import numpy as np
from pathlib import Path, PosixPath
import json
import copy
from tqdm import tqdm
import csv
from PIL import Image

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
                    try:
                        h_info = data['h_info']
                    except:
                        h_info = {'R': data["R"],
                                  'T': data["T"],
                                  'mtx': data["mtx"],
                                  'dist': data["dist"]}

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

def crop_image_PIL(image: Image, imagePoint, box):
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