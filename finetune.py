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


##############
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

    train_meta_dir = config["train_dataset"]["path"]
    train_shuffle = config["train_dataset"]["shuffle"]
    train_num_workers = config["train_dataset"]["num_workers"]
    test_meta_dir = config["test_dataset"]["path"]
    test_shuffle = config["test_dataset"]["shuffle"]
    test_num_workers = config["test_dataset"]["num_workers"]

    lr = config["train"]["lr"]
    num_epochs = config["train"]["num_epochs"]
    weight_decay = config["train"]["weight_decay"]
    train_batch_size = config["train"]["batch_size"]
    test_batch_size = config["test"]["batch_size"]

    if "lr_decay_step" in config["train"]:
        lr_decay_step = config["train"]["lr_decay_step"]

        if "lr_decay" in config["train"]:
            lr_decay = config["train"]["lr_decay"]
        else:
            lr_decay = None
    else:
        lr_decay_step = None


    ########## end loading config ##########

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    configs = {'lr': lr,
                'num_epochs': num_epochs,
                'weight_decay': weight_decay,
                'lr_decay_step': lr_decay_step,
                'lr_decay': lr_decay}


    # data processing
    # Define transformations
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }


    # Create datasets
    print(f"Using train dataset from {train_meta_dir}")
    print(f"Using test dataset from {test_meta_dir}")
    train_dataset = GraspDataset(train_meta_dir, transform=data_transforms['train'])
    val_dataset = GraspDataset(test_meta_dir, transform=data_transforms['val'])

    # Create WeightedRandomSampler
    # sample_weights = [train_dataset.class_weights[label] for label in train_dataset.labels]
    # sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)


    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=train_shuffle, num_workers=train_num_workers)
    val_loader = DataLoader(val_dataset, batch_size=test_batch_size, shuffle=test_shuffle, num_workers=test_num_workers)

    # Data loader dictionary
    dataloader = {
        'train': train_loader,
        'val': val_loader
    }

    # Dataset sizes
    dataset_sizes = {
        'train': len(train_dataset),
        'val': len(val_dataset)
    }

    print(f"Training dataset size: {dataset_sizes['train']}")
    print(f"Validation dataset size: {dataset_sizes['val']}")


    # model 
    print(f"Reading checkpoint from {model_ckpt_path}")
    model = GraspClssifier(model_ckpt_path)

    # distributed launch
    # model = nn.DataParallel(model)

    # finetune
    model, optimizer, loss = finetune(model, dataloader, dataset_sizes, configs, device, save_path)

    # save checkpoint
    save_checkpoint(model, optimizer, configs['num_epochs'], loss, f'{save_path}/ckpt.pth') 
    print(f"Checkpoint and validation result are saved to: {save_path}")


if __name__ == '__main__':
    ####### CHANGE this for different training ######
    main('/mnt/afs/huwensong/workspace/R3_grasp_classification/configs/config_train_0814_3.json')