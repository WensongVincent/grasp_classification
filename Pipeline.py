from PIL import Image
import cv2

import torch
from torch import nn
import torchvision
import torchaudio
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from GraspClassification import GraspClssifier, GraspDataset
from train import train, finetune, evaluation
from data_utils import board_to_world, world_to_image
from data_utils import crop_image_PIL as crop_image


class Pipeline():
    def __init__(self, ckpt_path = None, camera = None):
        if ckpt_path is None:
            raise ValueError('Need load checkpoint')
        self.model = GraspClssifier(ckpt_path).eval()

        if camera is None:
            raise ValueError('Need camera parameters')
        self.rvec = camera['R']
        self.tvec = camera['T']
        self.cameraMatrix = camera['cameraMatrix']
        self.distCoeffs = camera['distCoeffs']

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.z_in_mm = 78.0
        self.box = [160, 0, 0]

    @torch.no_grad()
    def __call__(self, image: Image, position):
        anchor = self.board_to_image(position)
        box = self.box
        image_crop = crop_image(image, anchor, box)

        image_crop_tensor = self.preprocess(image_crop).unsqueeze(0)        
        cls = self.model(image_crop_tensor).squeeze().argmax().item()
        return cls
    
    def board_to_image(self, position):
        x_in_mm, y_in_mm, z_in_mm = board_to_world(x=position[0], y=position[1], z_in_mm=self.z_in_mm)
        anchor = list(world_to_image(x_in_mm, y_in_mm, z_in_mm, self.rvec, self.tvec, self.cameraMatrix, self.distCoeffs))
        return anchor