from data_utils import board_to_world, world_to_image

import numpy as np
from pathlib import Path
import json

RT_PATH = '/mnt/afs/zhouzhijie/data/6.PieceGrasp'
K_PATH = '/mnt/afs/share_data/R3/v0.3/0.calibration/EVT1_Cam_TopR1+R2/Camera_02TopR1_H00027'

# Read intrinsics
intrinsics_path = Path(K_PATH)
intrinsics = intrinsics_path.rglob(r'*.json')
intrinsic = [intrinsic for intrinsic in intrinsics][0]

with open(intrinsic) as f:
    intrinsic = json.load(f)
cameraMatrix = np.array(intrinsic['mtx']).squeeze()
distCoeffs = np.array(intrinsic['dist']).squeeze()

# Read RT
rt_path = Path(RT_PATH)
rts = rt_path.rglob(r'*.json')

rvec_list = []
tvec_list = []
for rt in rts:
    with open(rt) as f:
        rt = json.load(f)
    rvec = np.array(rt['R']).squeeze()
    tvec = np.array(rt['T']).squeeze()
    rvec_list.append(rvec)
    tvec_list.append(tvec)
rvec_list = np.array(rvec_list)
tvec_list = np.array(tvec_list)
rvec = np.average(rvec_list, axis=0)
tvec = np.average(tvec_list, axis=0)

# Init position
nx, ny = (14, 14)
xs = np.linspace(-3, 10, nx)
ys = np.linspace(-3, 10, ny)
xv, yv = np.meshgrid(xs, ys)

zs = np.arange(57.0, 62.5, 0.5)


# Main 
image_point_grids_x = []
image_point_grids_y = []

for z in zs:
    image_point_grid_x = np.zeros((nx, ny))
    image_point_grid_y = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            x = int(xv[i, j])
            y = int(yv[i, j])
            position = np.array([x, y, z])
            x_world, y_world, z_world = board_to_world(x, y, z)
            image_point = world_to_image(x_world, y_world, z_world, rvec, tvec, cameraMatrix, distCoeffs)
            image_point_grid_x[i, j] = image_point[0]
            image_point_grid_y[i, j] = image_point[1]

    image_point_grids_x.append(image_point_grid_x)
    image_point_grids_y.append(image_point_grid_y)

image_point_grids_x = np.array(image_point_grids_x)
image_point_grids_y = np.array(image_point_grids_y)

image_point_diff_x = np.diff(image_point_grids_x, axis=0)
image_point_diff_y = np.diff(image_point_grids_y, axis=0)

image_point_diff_5mm_x = np.sum(image_point_diff_x, axis=0)
image_point_diff_5mm_y = np.sum(image_point_diff_y, axis=0)

print('')
