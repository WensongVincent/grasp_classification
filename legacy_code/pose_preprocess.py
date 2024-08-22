import json
import os
import numpy as np
from pathlib import Path
import csv

DATA_PATH = '/mnt/afs/huwensong/data/datasets/6.PieceGrasp/20240618_R3v0.3_6.PieceGrasp_P1/6.PieceGrasp/Data/NormalData/NormalLight/R3_EVT1_20240618/Pieces-Align/Occ-RoboticArm/Camera_01R2_G2704863'
states = ['Grabbed', 'Ungrabed']
Grabbed_pieces = ['B-Bishop', 'B-King', 'B-Knight', 'B-Pawn', 'B-Queen', 'B-Rook', 'W-Bishop', 'W-King', 'W-Knight', 'W-Pawn', 'W-Queen', 'W-Rook']
Ungrabed_pieces = ['Unknown-intersection']

# build position list
positions = []

nx, ny = (8, 8)
xs = np.linspace(0, 7, nx)
ys = np.linspace(0, 7, ny)
xv, yv = np.meshgrid(xs, ys)
for i in range(nx):
    for j in range(ny):
        x = xv[i, j]
        y = yv[i, j]
        position = [x, y]
        if (x == 5 and y == 0) or (x == 4 and y == 0):
            continue
        positions.append(position)



# modify position
for state in states:
    if state == 'Grabbed':
        for piece in Grabbed_pieces:
            data_path = Path(f'{DATA_PATH}/{state}/{piece}')
            # 62 positions in total
            poses = data_path.rglob(r'*.csv')
            for pose, position in zip(poses, positions):
                with open(pose, 'w', newline='') as writeFile:
                    writer = csv.writer(writeFile)
                    writer.writerow(position)
                # with open(pose, 'r', newline='') as readFile:
                #     reader = csv.reader(readFile)
                #     with open(pose, 'w', newline='') as writeFile:
                #         writer = csv.writer(writeFile)
                #         for row in reader:
                #             row[:2] = position
                #             writer.writerow(row)             

    elif state == 'Ungrabed':
        for piece in Ungrabed_pieces:
            data_path = Path(f'{DATA_PATH}/{state}/{piece}')
            # 62 positions in total
            poses = data_path.rglob(r'*.csv')
            for pose, position in zip(poses, positions):
                with open(pose, 'w', newline='') as writeFile:
                    writer = csv.writer(writeFile)
                    writer.writerow(position)


# for path_piece in path_pieces:
#     # Read images and poses
#     data_path = Path(f'{DATA_PATH}/{path_piece}')
#     images = data_path.rglob(r'*.png')

#     poss = []

#     # get xy grid
#     nx, ny = (8, 8)
#     xs = np.linspace(0, 7, nx)
#     ys = np.linspace(0, 7, ny)
#     xv, yv = np.meshgrid(xs, ys)

#     # get z
#     x1, x2, y1, y2 = (0, 7, 0, 7)
#     q11, q21, q12, q22 = (59.0, 57.71, 57.34, 57.79)

#     # store xy and z_in_mm in poses
#     for i in range(nx):
#         for j in range(ny):
#             x = int(xv[i, j])
#             y = int(yv[i, j])

#             # Bilinear interpolation
#             r1 = ((x2 - x) / (x2 - x1)) * q11 + ((x - x1) / (x2 - x1)) * q21
#             r2 = ((x2 - x) / (x2 - x1)) * q12 + ((x - x1) / (x2 - x1)) * q22
#             qxy = ((y2 - y) / (y2 - y1)) * r1 + ((y - y1) / (y2 - y1)) * r2

#             pos = [x, y, qxy]
            
#             # Append poses
#             if (x == 5 and y == 0) or (x == 4 and y == 0):
#                 continue

#             poss.append(pos)

#     poss.append([None, None, None])

#     # Write poses
#     for image, pos in zip(images, poss):
#         piece = image.parts[-3]
#         id = image.parts[-2]

#         if piece == 'calib-R2':
#             break
        
#         save_path = f'{DATA_PATH}/{piece}/{id}/position.json'

#         content = {'position': pos}

#         content = json.dumps(content, indent=4)
#         with open(save_path, 'w') as f:
#             f.write(content)
#         print(save_path, content)


    
