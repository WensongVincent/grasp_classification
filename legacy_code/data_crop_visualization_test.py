import os
import json
from pathlib import Path, PosixPath
from copy import copy

import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm

from data_utils import crop_image_PIL as crop_image

PROJECT_PATH = '/mnt/afs/huwensong/workspace/R3_grasp_classification'
OUTPUT_PATH = '/mnt/afs/huwensong/workspace/R3_grasp_classification/data_preprocessed/visualization'
VERSION = '0814_1' ###### CHANGE this for new metadata ######

retrive = False ###### CHANGE this for new metadata ######
retrive_list = ['1719974034222', '1719974169537', '1718680586339', '1719989983492', '1719990184447']

##############
# main function
def main():
    # Paths
    data_dir = f'{PROJECT_PATH}/meta_data'
    test_meta_dir = os.path.join(data_dir, f'metadata_{VERSION}_test.json')
    # test_meta_dir = os.path.join(data_dir, f'metadata_{VERSION}_test_gradcam.json')
    os.makedirs(f'{OUTPUT_PATH}/{VERSION}_test', exist_ok=True)

    # Create datasets
    image_paths = []
    labels = []
    anchors = []
    boxes = []

    content = []
    for meta_dir in [test_meta_dir]:
    # for meta_dir in [test_meta_dir]:
        with open(meta_dir, 'r') as file:
            content.append(json.load(file))
    content = [item for items in content for item in items]
    
    for item in content:
        image_paths.append(item['imagePath'])
        labels.append(item['gt'])
        anchors.append(item['anchor'])
        boxes.append(item['box'])


    for idx, _ in enumerate(tqdm(image_paths)):
        image_path = image_paths[idx]
        gt = labels[idx]
        anchor = anchors[idx]
        box = boxes[idx]

        time_stamp = PosixPath(image_path).parts[-2]
        
        if retrive:
            if time_stamp not in retrive_list:
                continue

        image = Image.open(image_path).convert("RGB")
        image_copy = copy(image)
        image = crop_image(image=image, imagePoint=anchor, box=box)
        image.save(f'{OUTPUT_PATH}/{VERSION}_test/{time_stamp}_{gt}.png')

        # image = cv2.cvtColor(np.array(image_copy), cv2.COLOR_RGB2BGR)
        # cv2.circle(image, anchor, 4, (0, 255, 0), -1)
        # cv2.circle(image, [anchor[0] + box[2], anchor[1] + box[1]], 4, (255, 0, 0), -1)
        # cv2.imwrite(f'{OUTPUT_PATH}/{VERSION}_test/{time_stamp}_{gt}_marked.png', image)

    


if __name__ == '__main__':
    main()
