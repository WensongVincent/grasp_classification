import os
import copy
import json
from collections import Counter
from pathlib import Path, PosixPath
import argparse

from PIL import Image
import numpy as np
import cv2

import torch
from torch import nn
import torchvision
import torchaudio
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from GraspClassification import GraspClssifier, GraspDataset
from utils import save_checkpoint, save_validation_results
from train import train, finetune, evaluation



# main function
def main(config_path):
    ########## loading config ##########
    config_path = Path(config_path)
    config = None
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    save_path = config["general"]["save_path"]

    model_num_classes = config["model"]["num_classes"]
    if "scale" in config["model"]:
        model_scale = config["model"]["scale"]
    if "ckpt_path" in config["model"]:
        model_ckpt_path = config["model"]["ckpt_path"]

    test_meta_dir = config["test_dataset"]["path"]
    shuffle = config["test_dataset"]["shuffle"]
    num_workers = config["test_dataset"]["num_workers"]

    batch_size = config["test"]["batch_size"]
    if "grad_cam" in config['test']:
        grad_cam = config['test']['grad_cam']

    ########## end loading config ##########

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # data processing
    # Define transformations
    data_transforms = {
        'val': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Create datasets
    print(f"Using dataset from {test_meta_dir}")
    val_dataset = GraspDataset(test_meta_dir, transform=data_transforms['val']) 

    # Create dataloaders
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    # Data loader dictionary
    dataloader = {
        'val': val_loader
    }

    # Dataset sizes
    dataset_sizes = {
        'val': len(val_dataset)
    }

    print(f"Validation dataset size: {dataset_sizes['val']}")

    # model 
    print(f"Reading checkpoint from {model_ckpt_path}")
    model = GraspClssifier(f'{model_ckpt_path}')
    
    # finetune
    evaluation(model, dataloader, dataset_sizes, device, save_path)
    print(f"Test result are saved to: {save_path}")


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="Read config file")
    # parser.add_argument('config_path', type=str, required=True, help="Path to config file")
    # args = parser.parse_args()
    
    # main(args.config_path)
    ####### CHANGE this for different testing ######
    main('/mnt/afs/huwensong/workspace/R3_grasp_classification/configs/config_test_0801_1_gradcam.json')