import os
import copy
import json
from collections import Counter
from pathlib import Path, PosixPath
import random
import string

from PIL import Image
import numpy as np
import cv2

import torch
from torch import nn
import torchvision
import torchaudio
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


def save_validation_results(results, txt_output_file, image_output_dir = None):
    output_parent_path = str(PosixPath(txt_output_file).parent)
    os.makedirs(output_parent_path, exist_ok=True)
    with open(txt_output_file, 'w') as f:
        for time_stamp, gt_label, predicted_label, image_path in results:
            if gt_label != predicted_label:
                f.write(f"{time_stamp}, GT: {gt_label}, Pred: {predicted_label}, ImagePath: {image_path} \n")
                # if image_output_dir is not None:
                #     #######
    

def save_checkpoint(model, optimizer, epoch, loss, file_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, file_path)
    print(f"Checkpoint saved")

def gradcam_in_train(model, inputs, image_paths, labels, anchors, boxes, result_path, save_gradcam=False, save_new_box=False):
    target_layers = [model.resnet18.layer4[-1]]
    input_tensor = inputs #.clone().detach()
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor)

    temp_path = f"{str(PosixPath(result_path).parent)}_visualization/{PosixPath(result_path).parts[-1]}/gradcam"
    os.makedirs(temp_path, exist_ok=True)

    for i in range(inputs.size(0)):
        grayscale_cam_ = grayscale_cam[i]
        if input_tensor.device.type == 'cuda':
            input_tensor_ = input_tensor[i].cpu()
        else:
            input_tensor_ = input_tensor[i]
        input_tensor_ = (input_tensor_ - input_tensor_.min()) / (input_tensor_.max() - input_tensor_.min())
        visualization = show_cam_on_image(input_tensor_.permute(1, 2, 0).numpy(), np.array(grayscale_cam_), use_rgb=True)
        if save_gradcam:
            cv2.imwrite(f'{temp_path}/{PosixPath(image_paths[i]).parts[-2]}.jpg', cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

        if save_new_box:
            b, _, _ = boxes[i]
            gt = labels[i]
            image_path = image_paths[i]
            x_a, y_a = anchors[i]
            x_a, y_a = (int(x_a), int(y_a))
            # # dot visualize here
            # temp_img = cv2.imread(image_path)
            # cv2.circle(temp_img, (x_a, y_a), 3, (255, 0, 0), -1)
            # cv2.rectangle(temp_img, (x_a-80, y_a-80), (x_a+80, y_a+80), (255, 0, 0,), 3)
            # # end visualize
            
            x_c, y_c = np.unravel_index(np.argmax(grayscale_cam_), grayscale_cam_.shape)
            x_c, y_c = (int(x_c * (b / 224)), int(y_c * (b / 224)))

            new_wh = 96

            x, y = (int(x_c + x_a - b / 2), int(y_c + y_a - b/2))
            # # dot visualize here
            # cv2.circle(temp_img, (x, y), 3, (0, 255, 0), -1)
            # # end visualize
            x_tl, y_tl = (int(x - new_wh/2), int(y - new_wh/2))
            # # box visualize here
            # cv2.rectangle(temp_img, (x_tl, y_tl), (x_tl+new_wh, y_tl+new_wh), (0, 255, 0), 3)
            # # end visualize
            # cv2.imwrite('/mnt/afs/huwensong/workspace/R3_grasp_classification/temp.jpg', temp_img)

            x_br, y_br = (int(x + new_wh/2), int(y + new_wh/2))

            content = to_sense_format([[x_tl, y_tl, x_br, y_br, str(gt.item())]], 1920, 1080)

            # content = {"width": 1920,
            #             "height": 1080,
            #             "valid": True,
            #             "rotate": 0,
            #             "step_1": {
            #                 "toolName": "rectTool",
            #                 "dataSourceStep": 0,
            #                 "result": [
            #                 {
            #                     "x": x_tl,
            #                     "y": y_tl,
            #                     "width": new_wh,
            #                     "height": new_wh,
            #                     "attribute": f"{gt}",
            #                     "valid": True,
            #                     "id": "",
            #                     "sourceID": "",
            #                     "textAttribute": "",
            #                     "order": 1
            #                 }
            #                 ]
            #             }
            #             }
            
            with open(f'{str(PosixPath(image_path).parent)}/image2.png.json', 'w') as f:
                json.dump(content, f, ensure_ascii=False, indent=4)

    outputs = cam.outputs
    return outputs


def randomID():
    return ''.join(random.choices(string.ascii_uppercase + string.ascii_lowercase + string.digits, k=8))


def to_sense_format(labelme_format, img_width, img_height, rotate=0):
    result = []
    for i, value in enumerate(labelme_format):
        result_i = {}
        x1, y1, x2, y2, label_str = value
        rect_width = round(abs(x2-x1), 4)
        rect_height = round(abs(y2-y1), 4)
        # if rect_width*rect_height < 1000: continue
        sense_label_str = label_str
        result_i = {
            "x": round(x1, 4), "y": round(y1, 4),
            "width": rect_width, "height": rect_height,
            "attribute": sense_label_str,
            "valid": True, "id": randomID(),
            "sourceID": "", "textAttribute": "", "order": i+1
        }
        result.append(result_i)
    
    sense_format = {"width": img_width, "height": img_height, "rotate": rotate, "valid": True,
        "step_1": {
            "dataSourceStep": 0,
            "toolName": "rectTool",
            "result":result
        }
    }
    return sense_format