import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
    # input("Press Enter to start training (set_memory_growth to True for all gpus)")

import os
import sys
import cv2
import numpy as np
from keras import Input, Model
from keras.callbacks import ModelCheckpoint

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_PATH) # чтобы из консольки можно было запускать


from another_unet import CreateModel
from utils_folder.logger import SaveModelEachBatch
from utils_folder.logger import TensorBoardBatchLogger
from unet import get_unet, build_model
from metrics import iou_loss_core
from generator import KerasGenerator
from utils_folder import config

config = config.CocoConfig()


class DataGen(tf.keras.utils.Sequence):
  
  def __init__(self , path_input , path_mask , batch_size = 8 , image_size = 128, flag=False):
    
    self.ids = sorted(os.listdir(path_input))   
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
train_gen = DataGen(path_input = "content/train2014" , path_mask = "content/mask_train_2014/" , batch_size = batch_size , image_size = image_size,  flag=True)
val_gen = DataGen(path_input =  "content/val2014", path_mask =  "content/mask_val_2014", batch_size = batch_size , image_size = image_size,   flag=True)
# Сетка 1:
# input_img = Input(img_size_target, name='img')
# model = get_unet(input_img, exit_channels=keras_gen.num_cats, n_filters=8, dropout=0.05, batchnorm=True)
# model.summary()

# Сетка 2:
# data_callback = DataCallback(path=PROJECT_PATH)
model_ = CreateModel(img_size_target=image_size)
model_.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy', iou_loss_core])
model_.summary()

# experiment name
name = "exp-000"
# path where tensorboard logs will be stored
log_dir = "experiments"

# create our custom logger
# logger = SummaryWriter(log_dir=osp.join(log_dir, name))
# log files will be saved in 'experiments/exp-000'
# Обучение:

# model_check = ModelCheckpoint('models/weights.{epoch:02d}-{loss:.5f}.hdf5', monitor='loss', verbose=0, save_best_only=True)
# print(model_check)

# log_dir = "logs/fit/{epoch:02d}-{loss:.5f}" 
logger = tf.keras.callbacks.TensorBoard(log_dir=log_dir,  update_freq='epoch', profile_batch=0,  histogram_freq=1)
# tf_logger = TensorBoardBatchLogger(project_path=PROJECT_PATH, batch_size=keras_gen.batch_size)
# save_model_batch = SaveModelEachBatch('models')
save_model_batch = SaveModelEachBatch('models23')
model_check = ModelCheckpoint('models23/weights.{epoch:02d}-{loss:.5f}.hdf5', monitor='loss', verbose=0, save_best_only=True)
train_steps =  len(os.listdir( "content/train2014"))/batch_size
history = model_.fit_generator(train_gen, 
                              validation_data = val_gen , 
                              steps_per_epoch= train_steps,
                              epochs=100,
                              callbacks=[ model_check, logger, save_model_batch])
