from data_preprocess_batch import board_to_world, world_to_image, read_jsonl_file

from pathlib import Path, PosixPath
import json
import numpy as np
from tqdm import tqdm
import csv
import cv2
import os


RT_PATH_LIST = ['/mnt/afs/huwensong/data/datasets/6.PieceGrasp_RT/20240618_R3v0.3_6.PieceGrasp_TestData_P1_hhh.jsonl',
                '/mnt/afs/huwensong/data/datasets/6.PieceGrasp_RT/20240703_R3v0.3_6.PieceGrasp_ClawIntersecting_TestData_P1_hhh.jsonl',
                '/mnt/afs/huwensong/data/datasets/6.PieceGrasp_RT/20240703_R3v0.3_6.PieceGrasp_FreeArea_TestData_P1_hhh.jsonl',
                '/mnt/afs/zhouzhijie/data/temp/20240722_R3v0.3_6.PieceGrasp_TestData_P2_hhh.jsonl']
K_PATH = '/mnt/afs/share_data/R3/v0.3/0.calibration/EVT1_Cam_TopR1+R2/Camera_01R2_G2704863'
OUTPUT_PATH = '/mnt/afs/huwensong/workspace/R3_grasp_classification/data_preprocessed/visualization'
OUTPUT_VERSION = '0729_3_grid_projection_experiment' ###### CHANGE this for new metadata ######


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

    rvec_list = []
    tvec_list = []

    anchor_list = []
    # Process the directories to metadata
    for image_path in tqdm(image_list, total=length):
        image_path_str = str(image_path)
        pos_path = list(Path(image_path_str).parent.rglob('*.csv'))[0]
        time_stamp = PosixPath(image_path).parts[-2]

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

        rvec_list.append(rvec)
        tvec_list.append(tvec)

        # read image
        image = cv2.imread(image_path_str, cv2.COLOR_BGR2RGB)
        
        # Init position
        nx, ny = (9, 9)
        xs = np.linspace(-0.5, 7.5, nx)
        ys = np.linspace(-0.5, 7.5, ny)
        xv, yv = np.meshgrid(xs, ys)

        # mark projected point on image
        for i in range(nx):
            for j in range(ny):
                x = xv[i, j]
                y = yv[i, j]
        
                # transformation
                x_in_mm, y_in_mm, z_in_mm = board_to_world(x, y, 0.0)
                anchor = list(world_to_image(x_in_mm, y_in_mm, z_in_mm, rvec, tvec, cameraMatrix, distCoeffs))
                anchor_list.append(anchor)
                cv2.circle(image, anchor, 4, (0, 255, 0), -1)

        os.makedirs(f'{OUTPUT_PATH}/{OUTPUT_VERSION}', exist_ok=True)
        cv2.imwrite(f'{OUTPUT_PATH}/{OUTPUT_VERSION}/{time_stamp}_projection_grid_marked_z00.png', image)




if __name__ == '__main__':
    main()