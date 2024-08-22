import torch
import torch.onnx
import numpy as np
from GraspClassification import GraspClssifier

@torch.no_grad()
def export_to_onnx(model_path, onnx_path, input_size):
    model = GraspClssifier(model_path).eval()
    dummy_input = torch.randn(input_size)#.numpy()
    input_names = ["inputs"]
    output_names = ["output"]
    torch.onnx.export(model, dummy_input, onnx_path, opset_version=10, verbose=True,
                    input_names=input_names, output_names=output_names)
    # torch.onnx.checker.check_model(onnx_path)  

export_to_onnx('/mnt/afs/huwensong/workspace/R3_grasp_classification/result/0729_2/ckpt.pth',
               '/mnt/afs/huwensong/workspace/R3_grasp_classification/result/0729_2/R3_GraspClassifier_0729_2.onnx',
               (1, 3, 224, 224))