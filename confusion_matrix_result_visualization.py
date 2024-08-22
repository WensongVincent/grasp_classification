import json
import os
from tqdm import tqdm
from PIL import Image

from data_utils import crop_image_PIL as crop_image

version = '0814_1'

#input
metadata_path = '/mnt/afs/huwensong/workspace/R3_grasp_classification/meta_data/metadata_0814_1_test.json'
result_path = f'/mnt/afs/huwensong/workspace/R3_grasp_classification/result/{version}/test_results.txt'
result_path_backup = f'/mnt/afs/huwensong/workspace/R3_grasp_classification/result/{version}/best_validation_results.txt'
# result_path_backup = f'/mnt/afs/huwensong/workspace/R3_grasp_classification/result/{version}/test_results.txt'

#output
visualization_dir = f'/mnt/afs/huwensong/workspace/R3_grasp_classification/result_visualization/{version}'
# visualization_dir = f'/mnt/afs/huwensong/workspace/R3_grasp_classification/result_visualization/{version}_test'
visual_falsePositive_dir = f'{visualization_dir}/falsePositive'
visual_falseNegative_dir = f'{visualization_dir}/falseNegative'

os.makedirs(visual_falsePositive_dir, exist_ok=True)
os.makedirs(visual_falseNegative_dir, exist_ok=True)

# read metadata
positive = 0
negative = 0
metadata = None
with open(metadata_path, 'r') as f:
    metadata = json.load(f)

for item in metadata:
    if item['gt'] == 0:
        negative += 1
    if item['gt'] == 1:
        positive += 1

total = positive + negative
print('Total', total)
print('Positive:', positive)
print('Negative:', negative)

metadata_map = {}
for item in metadata:
    metadata_map[item['imagePath']] = item

# read result
falsePositive = 0
falseNegative = 0
wrong = 0


if not os.path.exists(result_path):
    result_path = result_path_backup
with open(result_path, 'r') as f:
    for line in f:
        if len(line) < 2:
            continue
        wrong += 1
        GT = int(line[19])
        Pred = int(line[28])
        image_path = line[42:-2]
        time_stamp = line[:13]
        if GT == 1:
            falseNegative += 1
            image = Image.open(image_path).convert("RGB")
            anchor = metadata_map[image_path]['anchor']
            box = metadata_map[image_path]['box']
            image = crop_image(image=image, imagePoint=anchor, box=box)
            image.save(f'{visual_falseNegative_dir}/{time_stamp}_{GT}.png')
        elif GT == 0:
            falsePositive += 1
            image = Image.open(image_path).convert("RGB")
            anchor = metadata_map[image_path]['anchor']
            box = metadata_map[image_path]['box']
            image = crop_image(image=image, imagePoint=anchor, box=box)
            image.save(f'{visual_falsePositive_dir}/{time_stamp}_{GT}.png')
correct = total - wrong
print('Correct', correct)
print('Wrong', wrong)

print('False Positive', falsePositive)
print('False Negative', falseNegative)

truePositive = positive - falseNegative
trueNegative = negative - falsePositive

print('True Positive', truePositive)
print('True Negative', trueNegative)