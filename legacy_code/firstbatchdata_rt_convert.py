import json
from pathlib import Path
import os
from tqdm import tqdm

RT_PATH = '/mnt/afs/zhouzhijie/data/6.PieceGrasp'
IMAGE_PATH = '/mnt/afs/huwensong/data/datasets/6.PieceGrasp'
OUTPUT_PATH = '/mnt/afs/zhouzhijie/data/6.PieceGrasp'

def main():
    # Read RT
    rt_PATH = Path(RT_PATH)
    rt_paths = list(rt_PATH.rglob(r'*.json'))
    num = len(rt_paths)

    # Read images and build a dictionary
    image_PATH = Path(IMAGE_PATH)
    image_paths = list(image_PATH.rglob(r'*.png'))
    image_dict = {}

    for image_path in image_paths:
        time_stamp = image_path.parts[-2]
        if time_stamp not in image_dict:
            image_dict[time_stamp] = []
        image_dict[time_stamp].append(image_path)

    # Output file
    test_path = os.path.join(OUTPUT_PATH, '20240618_R3v0.3_6.PieceGrasp_TestData_P1_hhh.jsonl')
    train_path = os.path.join(OUTPUT_PATH, '20240618_R3v0.3_6.PieceGrasp_TrainData_P1_hhh.jsonl')
    test = []
    train = []

    # For each rt json file
    for rt_path in tqdm(rt_paths, total=num):
        save = {}
        h_info = None

        # Get rt json information
        with open(rt_path, 'r') as f:
            h_info = json.load(f)
        time_stamp = rt_path.parts[-2]

        # Find corresponding image
        found = False
        if time_stamp in image_dict:
            for image_path in image_dict[time_stamp]:
                imgpath = str(image_path)
                
                # Save to output jsonl
                save["imgpath"] = imgpath
                save["h_info"] = h_info
                
                if 'TestData' in image_path.parts:
                    test.append(save)
                elif 'TrainData' in image_path.parts:
                    train.append(save)
                else:
                    print('test or train not found')
                
                found = True
                break

        if not found:
            print(f'{time_stamp} not found')

    # Write into jsonl
    with open(test_path, 'w') as f:
        for entry in test:
            f.write(json.dumps(entry) + '\n')
    with open(train_path, 'w') as f:
        for entry in train:
            f.write(json.dumps(entry) + '\n')

if __name__ == '__main__':
    main()


# import json
# from pathlib import Path
# import os
# from tqdm import tqdm
# from copy import copy

# RT_PATH = '/mnt/afs/zhouzhijie/data/6.PieceGrasp'
# IMAGE_PATH = '/mnt/afs/huwensong/data/datasets/6.PieceGrasp'
# OUTPUT_PATH = '/mnt/afs/zhouzhijie/data/6.PieceGrasp'

# def main():
#     # Read RT
#     rt_PATH = Path(RT_PATH)
#     rt_paths = rt_PATH.rglob(r'*.json')
#     rt_paths = list(rt_paths)
#     num = len(rt_paths)

#     # Read images
#     image_PATH = Path(IMAGE_PATH)
#     image_paths = image_PATH.rglob(r'*.png')

#     # output file
#     test_path = os.path.join(OUTPUT_PATH, '20240618_R3v0.3_6.PieceGrasp_TestData_P1_hhh.jsonl')
#     train_path = os.path.join(OUTPUT_PATH, '20240618_R3v0.3_6.PieceGrasp_TrainData_P1_hhh.jsonl')
#     test = []
#     train = []
    
#     # for each rt json file
#     for rt_path in tqdm(rt_paths, total=num):
#         save = {}
#         h_info = None
#         imgpath = None

#         # Get rt json information
#         with open(rt_path, 'r') as f:
#             h_info = json.load(f)
#         time_stamp = rt_path.parts[-2]

#         # Find corresponding image 
#         found = False
#         # image_paths = image_PATH.rglob(r'*.png')
#         for image_path in image_paths:
#             # if found
#             if time_stamp in image_path.parts:
#                 imgpath = str(image_path)
#                 found = True
                
#                 # save to output jsonl
#                 save["imgpath"] = imgpath
#                 save["h_info"] = h_info
                
#                 if 'TestData' in image_path.parts:
#                     test.append(save)
#                 elif 'TrainData' in image_path.parts:
#                     train.append(save)
#                 else:
#                     print('test or train not found')
                
#                 break


#         if found == False:
#             print(f'{time_stamp} not found')

#     # Write into jsonl
#     with open(test_path, 'w') as f:
#         for entry in test:
#             f.write(json.dumps(entry) + '\n')
#     with open(train_path, 'w') as f:
#         for entry in train:
#             f.write(json.dumps(entry) + '\n')

        

        


# if __name__ == '__main__':
#     main()