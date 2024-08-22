# R3_GraspClassification

## Code notes
GraspClassification.py --- Model definition and Dataset Definition

train.py --- train, validation and test core logics

finetune.py --- model training given config file, save ckpt and bad case result

test.py --- model testing given config file, save bad case result

data_preprocess_batch.py --- Metadata generation from .jsonl file (include image_path, camera intrinsics and extrinsics, provided by Zhijie)

data_preprocess_labelled.py --- Metadata generation from .jsonl and annotations (provided by Zhaojun after GradCam pre-annotation)

confusion_matrix_result_visualization.py --- read bad case result and culculate confusion matrix

Pipeline.py --- pipeline class definition

test_pipeline --- pipeline testing

## Workflow
See file: https://ones.ainewera.com/wiki/#/team/JNwe8qUX/space/UrgpTLTa/page/DduZRXbR
