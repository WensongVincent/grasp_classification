import os
import copy
import json
from collections import Counter
from pathlib import Path, PosixPath

from PIL import Image
import numpy as np
import cv2

import torch
from torch import nn
import torchvision
import torchaudio
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from collections import OrderedDict

from data_utils import crop_image_PIL as crop_image

#############
# model define
class GraspClssifier_(nn.Module):
    def __init__(self, scale=0.25, num_classes=2, pretrained=None):
        super().__init__()

        # Load the pretrained ResNet-18 model
        original_resnet = models.resnet18(pretrained=None)
        
        # Modify the original architecture by reducing the intermediate channels
        new_conv1_out_channels = int(original_resnet.conv1.out_channels * scale)
        self.conv1 = nn.Conv2d(3, new_conv1_out_channels, kernel_size=7, stride=2, padding=3, bias=False)  # Change number of output channels
        self.bn1 = nn.BatchNorm2d(new_conv1_out_channels)
        self.relu = original_resnet.relu
        self.maxpool = original_resnet.maxpool

        self.layer1 = self._modify_layer(original_resnet.layer1, scale)
        self.layer2 = self._modify_layer(original_resnet.layer2, scale)
        self.layer3 = self._modify_layer(original_resnet.layer3, scale)
        self.layer4 = self._modify_layer(original_resnet.layer4, scale)
        
        self.avgpool = original_resnet.avgpool
        new_fc_in_channels = int(original_resnet.fc.in_features * scale)
        self.fc = nn.Linear(new_fc_in_channels, num_classes)  # Adjust the input features

        # Load pretrained weights if specified
        if pretrained == None:
            # Initialize weights
            self._initialize_weights()
        elif pretrained == 'default':
            self._load_pretrained_weights(pretrained)
        else:
            self._load_pretrained_weights(pretrained)

    def _modify_layer(self, layer, scale):
        # Modify each layer to reduce the number of channels
        new_layer = []
        for block in layer:
            new_in_channels = int(block.conv1.in_channels * scale)
            new_out_channels = int(block.conv1.out_channels * scale)
            new_block = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(new_in_channels, new_out_channels, kernel_size=3, stride=block.conv1.stride, padding=1, bias=False)),
                ('bn1', nn.BatchNorm2d(new_out_channels)),
                ('relu', block.relu),
                ('conv2', nn.Conv2d(new_out_channels, new_out_channels, kernel_size=3, padding=1, bias=False)),
                ('bn2', nn.BatchNorm2d(new_out_channels))
            ]))
            new_layer.append(new_block)
        return nn.Sequential(*new_layer)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _load_pretrained_weights(self, ckpt_path=None):
        pretrained_dict = None

        if ckpt_path == None:
            raise("ckpt should not be None")
        
        elif ckpt_path == "default":
            resenet18_default = models.resnet18(pretrained=models.ResNet18_Weights.DEFAULT)
            pretrained_dict = resenet18_default.state_dict()
        
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(ckpt_path, device)["model_state_dict"]

        custom_model_dict = self.state_dict()

        # Filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in custom_model_dict and v.size() == custom_model_dict[k].size()}

        # Overwrite entries in the existing state dict
        custom_model_dict.update(pretrained_dict)

        # Load the new state dict
        self.load_state_dict(custom_model_dict)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        y_pred = self.fc(x)

        return y_pred
    

class GraspClssifier(nn.Module):
    def __init__(self, ckpt_path = None):
        super().__init__()
        if ckpt_path is None or ckpt_path == "default":
            weights = models.ResNet18_Weights.DEFAULT
            self.resnet18 = models.resnet18(weights=weights)
        else:
            self.resnet18 = models.resnet18(weights=None)
        
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, 2)

        if ckpt_path is not None and ckpt_path != "default":
            self.load_checkpoint_from_path(ckpt_path)

    def forward(self, x):
        y_pred = self.resnet18(x)
        return y_pred
    
    def load_checkpoint_from_path(self, path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ckpt = torch.load(path, map_location=device)
        self.load_state_dict(ckpt['model_state_dict'])


class GraspDataset(Dataset):
    def __init__(self, meta_dir, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.anchors = []
        self.boxes = []

        content = None
        with open(meta_dir, 'r') as file:
            content = json.load(file)
        
        for item in content:
            self.image_paths.append(item['imagePath'])
            self.labels.append(item['gt'])
            self.anchors.append(item['anchor'])
            self.boxes.append(item['box'])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        gt = self.labels[idx]
        anchor = self.anchors[idx]
        box = self.boxes[idx]

        image = Image.open(image_path).convert("RGB")
        image = crop_image(image=image, imagePoint=anchor, box=box)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(gt, dtype=torch.long), image_path, torch.tensor(anchor, dtype=torch.int), torch.tensor(box, dtype=torch.int)

