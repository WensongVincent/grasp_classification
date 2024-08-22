import os
import shutil
import random

# Modify the data file structure
'''
Modify from:
data/
    cropped/
        class1/
        class2/
    gt/
        class1/
        class2/

To:
data/
    train/
        image/
            class1/
            class2/
        gt/
            class1/
            class2/
    val/
        image/
            class1/
            class2/
        gt/
            class1/
            class2/
'''

def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def split_dataset(image_source_dir, gt_source_dir, image_dest_train_dir, image_dest_val_dir, gt_dest_train_dir, gt_dest_val_dir, split_ratio=0.8):
    for class_name in os.listdir(image_source_dir):
        image_class_dir = os.path.join(image_source_dir, class_name)
        gt_class_dir = os.path.join(gt_source_dir, class_name)

        if os.path.isdir(image_class_dir) and os.path.isdir(gt_class_dir):
            files = os.listdir(image_class_dir)
            random.shuffle(files)
            
            split_index = int(len(files) * split_ratio)
            train_files = files[:split_index]
            val_files = files[split_index:]

            train_image_class_dir = os.path.join(image_dest_train_dir, class_name)
            val_image_class_dir = os.path.join(image_dest_val_dir, class_name)
            train_gt_class_dir = os.path.join(gt_dest_train_dir, class_name)
            val_gt_class_dir = os.path.join(gt_dest_val_dir, class_name)
            
            create_dir(train_image_class_dir)
            create_dir(val_image_class_dir)
            create_dir(train_gt_class_dir)
            create_dir(val_gt_class_dir)

            for file in train_files:
                shutil.copy(os.path.join(image_class_dir, file), os.path.join(train_image_class_dir, file))
                gt_file = os.path.splitext(file)[0] + '.json'
                shutil.copy(os.path.join(gt_class_dir, gt_file), os.path.join(train_gt_class_dir, gt_file))
                
            for file in val_files:
                shutil.copy(os.path.join(image_class_dir, file), os.path.join(val_image_class_dir, file))
                gt_file = os.path.splitext(file)[0] + '.json'
                shutil.copy(os.path.join(gt_class_dir, gt_file), os.path.join(val_gt_class_dir, gt_file))

# Paths
data_dir = '/mnt/afs/huwensong/workspace/R3_grasp_classification/data_preprocessed'
image_dir = os.path.join(data_dir, 'cropped')
gt_dir = os.path.join(data_dir, 'gt')

train_image_dir = os.path.join(data_dir, 'train', 'image')
val_image_dir = os.path.join(data_dir, 'val', 'image')

train_gt_dir = os.path.join(data_dir, 'train', 'gt')
val_gt_dir = os.path.join(data_dir, 'val', 'gt')

# Create train and val directories
create_dir(train_image_dir)
create_dir(val_image_dir)
create_dir(train_gt_dir)
create_dir(val_gt_dir)

# Split the dataset
split_dataset(image_dir, gt_dir, train_image_dir, val_image_dir, train_gt_dir, val_gt_dir, split_ratio=0.8)

print("Dataset split completed.")