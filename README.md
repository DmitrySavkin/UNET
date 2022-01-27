# Multiclass Segmentation using Unet for COCO dataset

Repository for training Neural Network for Multiclass task (80 classes) for coco dataset

## Fast Launch instructions:
Run init.py for downloading all needed data (annotations + images)
- main/train.py for training
- main/predict.py for prediction
- examples/example_load_img_and_mask.py for visualization 

## Installation using anaconda and pip

```bash
conda create -n coco_unet_tensorflow python=3.8
conda activate coco_unet_tensorflow
conda install tensorflow cudatoolkit=11
pip install -r requirements.txt # will skip tensorflow
```
