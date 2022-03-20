from keras.models import load_model
import os
import sys
import random
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print((len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs"))
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
    eval(input("Press Enter to start training (set_memory_growth to True for all gpus)"))

import os
import sys
from keras import Input, Model
from keras.callbacks import ModelCheckpoint

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_PATH) # чтобы из консольки можно было запускать


import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *

from another_unet import CreateModel
from utils_folder.logger import SaveModelEachBatch
from utils_folder.logger import TensorBoardBatchLogger
from unet import get_unet, build_model
from metrics import iou_loss_core
from generator import KerasGenerator
from utils_folder import config

seed = 2019

random.seed = seed
np.random.seed = seed 
class DataGen(tf.keras.utils.Sequence):
  
  def __init__(self , path_input , path_mask , batch_size = 8 , image_size = 128):
    
    self.ids = os.listdir(path_input)
    self.path_input = path_input
    self.path_mask = path_mask
    self.batch_size = batch_size
    self.image_size = image_size
    self.on_epoch_end()
  
  def __load__(self , id_name):
    
    image_path = os.path.join(self.path_input , id_name)
    mask_path = os.path.join(self.path_mask , id_name) 
    
    image = cv2.imread(image_path , 1) # 1 specifies RGB format
    image = cv2.resize(image , (self.image_size , self.image_size)) # resizing before inserting to the network
    
    mask = cv2.imread(mask_path , -1)
    # print("Mask ", mask , (self.image_size , self.image_size), "Mask path " , mask_path)
    # input("Enter")
    mask = cv2.resize(mask , (self.image_size , self.image_size))
    mask = mask.reshape((self.image_size , self.image_size , 1))
      
    #normalize image
    image = image / 255.0
    mask = mask / 255.0
    
    return image , mask
  
  def __getitem__(self , index):
    
    if (index + 1)*self.batch_size > len(self.ids):
      self.batch_size = len(self.ids) - index * self.batch_size
        
    file_batch = self.ids[index * self.batch_size : (index + 1) * self.batch_size]
    
    images = []
    masks = []
    
    for id_name in file_batch : 
      
      _img , _mask = self.__load__(id_name)
      images.append(_img)
      masks.append(_mask)
    
    
    images = np.array(images)
    masks = np.array(masks)
    
    
    return images , masks
  
  
  def on_epoch_end(self):
    pass
  
  
  def __len__(self):
    
    return int(np.ceil(len(self.ids) / float(self.batch_size)))



image_size = 128
epochs = 100
batch_size = 1
model_file = '../../models23/weights.99-0.02342.hdf5'
val_gen = DataGen(path_input =  "../../content/val2014", path_mask =  "../../content/mask_val_2014", batch_size = batch_size , image_size = image_size)
from tensorflow import keras
from metrics import iou_loss_core
model = keras.models.load_model(model_file, custom_objects={'iou_loss_core': iou_loss_core})
model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy')

x, y = val_gen.__getitem__(4)
result = model.predict(x)

result = result > 0.5



