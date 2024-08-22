import os
import copy
import json
from collections import Counter
from pathlib import Path, PosixPath
from tqdm import tqdm

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
from utils import save_checkpoint, save_validation_results, gradcam_in_train



# training code
def train(model: nn.Module, 
          dataloader: DataLoader, 
          dataset_sizes, 
          criterion, 
          optimizer, 
          device, 
          num_epochs=10,
          result_path=None,
          scheduler=None,
          grad_cam=False):
    
    if result_path is None:
        raise ValueError('Result path should not be None')

    best_acc = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    if num_epochs == 0:
        model.eval()
        running_corrects = 0

        # Collect validation results
        val_results = []
        val_results_timestamp = []

        # Iterate over data.
        for inputs, labels, image_paths, anchors, boxes in tqdm(dataloader['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward
            # grad_cam
            if grad_cam:
                outputs = gradcam_in_train(model=model, inputs=inputs, image_paths=image_paths,
                                            labels=labels, anchors=anchors, boxes=boxes, 
                                            result_path=result_path, save_gradcam=True, save_new_box=True)
            # original
            else:
                outputs = model(inputs)
            
            _, preds = torch.max(outputs, 1)

            # Collect results
            for i in range(inputs.size(0)):
                time_stamp = PosixPath(image_paths[i]).parts[-2]
                predicted_label = preds[i].item()
                gt_label = labels[i].item()
                if time_stamp not in val_results_timestamp:
                    val_results.append((time_stamp, gt_label, predicted_label, image_paths[i]))
                    val_results_timestamp.append(time_stamp)

            # statistics
            running_corrects += torch.sum(preds == labels.data)

        best_acc = running_corrects.double() / dataset_sizes['val']

        # Save validation results
        save_validation_results(val_results, f'{result_path}/test_results.txt')

        epoch_loss = None
    
    else:
        # looping epochs
        for epoch in range(num_epochs):
            print(f'Epoch {epoch} / {num_epochs - 1}')
            print('-' * 10)
            
            # training and validation
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                # Collect validation results
                val_results = []
                val_results_timestamp = []

                # Iterate over data
                for inputs, labels, image_paths, _, _ in tqdm(dataloader[phase]):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the gradient in gradient flow
                    optimizer.zero_grad()
                    
                    # forward pass
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backwards + backprop optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    
                        # Collect results if in validation phase
                        if phase == 'val':
                            for i in range(inputs.size(0)):
                                time_stamp = PosixPath(image_paths[i]).parts[-2]
                                predicted_label = preds[i].item()
                                gt_label = labels[i].item()
                                if time_stamp not in val_results_timestamp:
                                    val_results.append((time_stamp, gt_label, predicted_label, image_paths[i]))
                                    val_results_timestamp.append(time_stamp)
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

                # Save validation results
                # if phase == 'val':
                #     save_validation_results(val_results, f'{RESULT_PATH}/epoch{epoch}_validation_results.txt')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    save_validation_results(val_results, f'{result_path}/best_validation_results.txt')
            
            # update learning rate
            if scheduler is not None:
                scheduler.step()

    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    if num_epochs > 0:
        model.load_state_dict(best_model_wts)
    return model, epoch_loss


def finetune(model: nn.Module, dataloader: DataLoader, dataset_sizes, configs, device, result_path=None):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=configs['lr'], weight_decay=configs['weight_decay'])
    if 'lr_decay_step' in configs:
        if 'lr_decay' in configs:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=configs['lr_decay_step'], gamma=configs['lr_decay'])
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=configs['lr_decay_step'])
    else:
        scheduler = None
    model, loss = train(model, dataloader, dataset_sizes, criterion, optimizer, device, configs['num_epochs'], result_path, scheduler)
    return model, optimizer, loss


def evaluation(model: nn.Module, dataloader: DataLoader, dataset_sizes, device, result_path=None, grad_cam=False):
    model = model.to(device)
    _, _ = train(model, dataloader, dataset_sizes, criterion=None, optimizer=None, device=device, num_epochs=0, result_path=result_path, grad_cam=grad_cam)
