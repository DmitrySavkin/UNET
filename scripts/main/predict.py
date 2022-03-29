import cv2
import numpy as np
import os
import sys

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_PATH) # чтобы из консольки можно было запускать


from utils_folder.prediction_util import custom_version
from datetime import date, datetime
from keras.models import load_model
from generator import KerasGenerator
from matplotlib import pyplot as plt
from metrics import iou_loss_core
from pycocotools.coco import COCO
import skimage.io as io




ANNOTATIONS_FILE = 'coco_dataset/annotations/instances_val2017.json'
'''
PLEASE USE models23 if nothing works
'''
MODEL_PATH = "models23/weights.08-0.30780.hdf5" # '../../models/weights.04-0.01406.hdf5'


 
coco = COCO(ANNOTATIONS_FILE)

def get_class_name(class_id):
    pass 

def predict(model_file, index):
    from keras.metrics import MeanIoU 
    pass



if __name__ == '__main__':
    import os
    arr = os.listdir("models26")
  
  
  
    #custom_version(MODEL_PATH, '../../content/train2014/COCO_train2014_000000000061.jpg', output='result_18.03.2022.jpg') 
  
  
  
    i = 0
    for file in arr:
        if file.endswith(".hdf5"): 
            print(file,  i)
            custom_version(f"models26/{file}", 'scripts/main/4.jpg', output = f'scripts/main/result/result_{file}_{date.today()}_{datetime.now()}.png')
            i += 1
    #pass
    

  