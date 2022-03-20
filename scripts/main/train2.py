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


def down_block(
    input_tensor,
    no_filters,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding="same",
    kernel_initializer="he_normal",
    max_pool_window=(2, 2),
    max_pool_stride=(2, 2)
):
    conv = Conv2D(
        filters=no_filters,
        kernel_size=kernel_size,
        strides=strides,
        activation=None,
        padding=padding,
        kernel_initializer=kernel_initializer
    )(input_tensor)

    conv = BatchNormalization(scale=True)(conv)

    conv = Activation("relu")(conv)

    conv = Conv2D(
        filters=no_filters,
        kernel_size=kernel_size,
        strides=strides,
        activation=None,
        padding=padding,
        kernel_initializer=kernel_initializer
    )(conv)

    conv = BatchNormalization(scale=True)(conv)

    # conv for skip connection
    conv = Activation("relu")(conv)

    pool = MaxPooling2D(pool_size=max_pool_window, strides=max_pool_stride)(conv)

    return conv, pool


def bottle_neck(
    input_tensor,
    no_filters,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding="same",
    kernel_initializer="he_normal"
):
    conv = Conv2D(
        filters=no_filters,
        kernel_size=kernel_size,
        strides=strides,
        activation=None,
        padding=padding,
        kernel_initializer=kernel_initializer
    )(input_tensor)

    conv = BatchNormalization(scale=True)(conv)

    conv = Activation("relu")(conv)

    conv = Conv2D(
        filters=no_filters,
        kernel_size=kernel_size,
        strides=strides,
        activation=None,
        padding=padding,
        kernel_initializer=kernel_initializer
    )(conv)

    conv = BatchNormalization(scale=True)(conv)

    conv = Activation("relu")(conv)

    return conv
def up_block(    
    input_tensor,
    no_filters,
    skip_connection, 
    kernel_size=(3, 3),
    strides=(1, 1),
    upsampling_factor = (2,2),
    max_pool_window = (2,2),
    padding="same",
    kernel_initializer="he_normal"):
    
    
    conv = Conv2D(
        filters = no_filters,
        kernel_size= max_pool_window,
        strides = strides,
        activation = None,
        padding = padding,
        kernel_initializer=kernel_initializer
    )(UpSampling2D(size = upsampling_factor)(input_tensor))
    
    conv = BatchNormalization(scale=True)(conv)

    conv = Activation("relu")(conv) 
    
    
    conv = concatenate( [skip_connection , conv]  , axis = -1)
    
    
    conv = Conv2D(
        filters=no_filters,
        kernel_size=kernel_size,
        strides=strides,
        activation=None,
        padding=padding,
        kernel_initializer=kernel_initializer
    )(conv)

    conv = BatchNormalization(scale=True)(conv)

    conv = Activation("relu")(conv)

    conv = Conv2D(
        filters=no_filters,
        kernel_size=kernel_size,
        strides=strides,
        activation=None,
        padding=padding,
        kernel_initializer=kernel_initializer
    )(conv)

    conv = BatchNormalization(scale=True)(conv)

    conv = Activation("relu")(conv)
    
    return conv
def output_block(input_tensor,
    padding="same",
    kernel_initializer="he_normal"
):
    
    conv = Conv2D(
        filters=2,
        kernel_size=(3,3),
        strides=(1,1),
        activation="relu",
        padding=padding,
        kernel_initializer=kernel_initializer
    )(input_tensor)
    
    
    conv = Conv2D(
        filters=1,
        kernel_size=(1,1),
        strides=(1,1),
        activation="sigmoid",
        padding=padding,
        kernel_initializer=kernel_initializer
    )(conv)
    
    
    return conv

def UNet(input_shape = (128,128,3)):
    
    filter_size = [64,128,256,512,1024]
    
    inputs = Input(shape = input_shape)
    
    d1 , p1 = down_block(input_tensor= inputs,
                         no_filters=filter_size[0],
                         kernel_size = (3,3),
                         strides=(1,1),
                         padding="same",
                         kernel_initializer="he_normal",
                         max_pool_window=(2,2),
                         max_pool_stride=(2,2))
    
    
    d2 , p2 = down_block(input_tensor= p1,
                         no_filters=filter_size[1],
                         kernel_size = (3,3),
                         strides=(1,1),
                         padding="same",
                         kernel_initializer="he_normal",
                         max_pool_window=(2,2),
                         max_pool_stride=(2,2))
    
    
    
    d3 , p3 = down_block(input_tensor= p2,
                         no_filters=filter_size[2],
                         kernel_size = (3,3),
                         strides=(1,1),
                         padding="same",
                         kernel_initializer="he_normal",
                         max_pool_window=(2,2),
                         max_pool_stride=(2,2))
    
    
    
    d4 , p4 = down_block(input_tensor= p3,
                         no_filters=filter_size[3],
                         kernel_size = (3,3),
                         strides=(1,1),
                         padding="same",
                         kernel_initializer="he_normal",
                         max_pool_window=(2,2),
                         max_pool_stride=(2,2))
    
    
    b = bottle_neck(input_tensor= p4,
                         no_filters=filter_size[4],
                         kernel_size = (3,3),
                         strides=(1,1),
                         padding="same",
                         kernel_initializer="he_normal")
    
    
    
    u4 = up_block(input_tensor = b,
                  no_filters = filter_size[3],
                  skip_connection = d4,
                  kernel_size=(3, 3),
                  strides=(1, 1),
                  upsampling_factor = (2,2),
                  max_pool_window = (2,2),
                  padding="same",
                  kernel_initializer="he_normal")
    
    u3 = up_block(input_tensor = u4,
                  no_filters = filter_size[2],
                  skip_connection = d3,
                  kernel_size=(3, 3),
                  strides=(1, 1),
                  upsampling_factor = (2,2),
                  max_pool_window = (2,2),
                  padding="same",
                  kernel_initializer="he_normal")
    
    
    u2 = up_block(input_tensor = u3,
                  no_filters = filter_size[1],
                  skip_connection = d2,
                  kernel_size=(3, 3),
                  strides=(1, 1),
                  upsampling_factor = (2,2),
                  max_pool_window = (2,2),
                  padding="same",
                  kernel_initializer="he_normal")
    
    
    u1 = up_block(input_tensor = u2,
                  no_filters = filter_size[0],
                  skip_connection = d1,
                  kernel_size=(3, 3),
                  strides=(1, 1),
                  upsampling_factor = (2,2),
                  max_pool_window = (2,2),
                  padding="same",
                  kernel_initializer="he_normal")
    
    
    
    output = output_block(input_tensor=u1 , 
                         padding = "same",
                         kernel_initializer= "he_normal")
    
    model = Model(inputs = inputs , outputs = output)
    
    
    return model


def get_customer_labels():
    import math
    from random import randrange
    d =  {0: 'unlabeled',
         1: 'person',
         2: 'bicycle',
         3: 'car',
         4: 'motorcycle',
         5: 'airplane',
         6: 'bus',
         7: 'train',
         8: 'truck',
         9: 'boat',
         10: 'traffic light',
         11: 'fire hydrant',
         12: 'street sign',
         13: 'stop sign',
         14: 'parking meter',
         15: 'bench',
         16: 'bird',
         17: 'cat',
         18: 'dog',
         19: 'horse',
         20: 'sheep',
         21: 'cow',
         22: 'elephant',
         23: 'bear',
         24: 'zebra',
         25: 'giraffe',
         26: 'hat',
         27: 'backpack',
         28: 'umbrella',
         29: 'shoe',
         30: 'eye glasses',
         31: 'handbag',
         32: 'tie',
         33: 'suitcase',
         34: 'frisbee',
         35: 'skis',
         36: 'snowboard',
         37: 'sports ball',
         38: 'kite',
         39: 'baseball bat',
         40: 'baseball glove',
         41: 'skateboard',
         42: 'surfboard',
         43: 'tennis racket',
         44: 'bottle',
         45: 'plate',
         46: 'wine glass',
         47: 'cup',
         48: 'fork',
         49: 'knife',
         50: 'spoon',
         51: 'bowl',
         52: 'banana',
         53: 'apple',
         54: 'sandwich',
         55: 'orange',
         56: 'broccoli',
         57: 'carrot',
         58: 'hot dog',
         59: 'pizza',
         60: 'donut',
         61: 'cake',
         62: 'chair',
         63: 'couch',
         64: 'potted plant',
         65: 'bed',
         66: 'mirror',
         67: 'dining table',
         68: 'window',
         69: 'desk',
         70: 'toilet',
         71: 'door',
         72: 'tv',
         73: 'laptop',
         74: 'mouse',
         75: 'remote',
         76: 'keyboard',
         77: 'cell phone',
         78: 'microwave',
         79: 'oven',
         80: 'toaster',
         81: 'sink',
         82: 'refrigerator',
         83: 'blender',
         84: 'book',
         85: 'clock',
         86: 'vase',
         87: 'scissors',
         88: 'teddy bear',
         89: 'hair drier',
         90: 'toothbrush',
         91: 'hair brush',  # Last class of Thing
         92: 'banner',  # Beginning of Stuff
         93: 'blanket',
         94: 'branch',
         95: 'bridge',
         96: 'building-other',
         97: 'bush',
         98: 'cabinet',
         99: 'cage',
         100: 'cardboard',
         101: 'carpet',
         102: 'ceiling-other',
         103: 'ceiling-tile',
         104: 'cloth',
         105: 'clothes',
         106: 'clouds',
         107: 'counter',
         108: 'cupboard',
         109: 'curtain',
         110: 'desk-stuff',
         111: 'dirt',
         112: 'door-stuff',
         113: 'fence',
         114: 'floor-marble',
         115: 'floor-other',
         116: 'floor-stone',
         117: 'floor-tile',
         118: 'floor-wood',
         119: 'flower',
         120: 'fog',
         121: 'food-other',
         122: 'fruit',
         123: 'furniture-other',
         124: 'grass',
         125: 'gravel',
         126: 'ground-other',
         127: 'hill',
         128: 'house',
         129: 'leaves',
         130: 'light',
         131: 'mat',
         132: 'metal',
         133: 'mirror-stuff',
         134: 'moss',
         135: 'mountain',
         136: 'mud',
         137: 'napkin',
         138: 'net',
         139: 'paper',
         140: 'pavement',
         141: 'pillow',
         142: 'plant-other',
         143: 'plastic',
         144: 'platform',
         145: 'playingfield',
         146: 'railing',
         147: 'railroad',
         148: 'river',
         149: 'road',
         150: 'rock',
         151: 'roof',
         152: 'rug',
         153: 'salad',
         154: 'sand',
         155: 'sea',
         156: 'shelf',
         157: 'sky-other',
         158: 'skyscraper',
         159: 'snow',
         160: 'solid-other',
         161: 'stairs',
         162: 'stone',
         163: 'straw',
         164: 'structural-other',
         165: 'table',
         166: 'tent',
         167: 'textile-other',
         168: 'towel',
         169: 'tree',
         170: 'vegetable',
         171: 'wall-brick',
         172: 'wall-concrete',
         173: 'wall-other',
         174: 'wall-panel',
         175: 'wall-stone',
         176: 'wall-tile',
         177: 'wall-wood',
         178: 'water-other',
         179: 'waterdrops',
         180: 'window-blind',
         181: 'window-other',
         182: 'wood'}
    X = [(r, g, b)  for b in range(100, 256, 1) for g in range(100,256, 1)  for r in range(100, 256, 1) ] 
    result = dict()  
    for keys, val in list(d.items()):
        x = X[randrange(len(X))]
        temp = {"id": keys, "category": val, "color": x}
        result[keys] = temp
        X.remove(x)
    # print(result)
    return result







def custom_version(model, image_name):
    customer_labels = get_customer_labels()
    print((customer_labels[5]))
    original = cv2.imread(image_name)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
  
    x_train = np.expand_dims(original, 0)
     
    print((x_train.shape, x_train.min(), x_train.max(), x_train.dtype))
    y_pred = model.predict(x_train)
    y_max_image = np.argmax(y_pred, axis=3)[0, :, :]
    print((np.nonzero(y_max_image)))
    # image_id = np.array(keras_gen.batch_train_indecies[0])
    np.save('y_pred.npy', y_pred)
    np.save('x_test.npy', x_train)
    #  result_img = np.zeros(y_max_image.shape, dtype=int)
    h, w = y_max_image.shape 
    img_result = np.zeros((h, w, 3), dtype=int)
    for  y in range(h):
        for x in range(w):
            id_class = y_max_image[ y, x]
            if id_class in customer_labels:
                img_result[y, x , 0] = customer_labels[id_class]['color'][0]
                img_result[y, x , 1] = customer_labels[id_class]['color'][1]
                img_result[y, x , 2] = customer_labels[id_class]['color'][2]
    # np.save('image_id.npy', image_id)
    
    
    print(y_max_image)
    print((y_max_image.shape, x_train.shape))
    print((np.unique(np.array(y_max_image), return_counts=True)))
    for cl in np.unique(np.array(y_max_image)):
        print((customer_labels[cl]['category']))
    cv2.imwrite('result.png', img_result)


log_dir = "experiments"
from metrics import iou_loss_core

model = UNet(input_shape = (128,128, 3))
from metrics import iou_loss_core
model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy', iou_loss_core ])
image_size = 128
epochs = 30
batch_size = 1
train_gen = DataGen(path_input = "content/train2014" , path_mask = "content/mask_train_2014/" , batch_size = batch_size , image_size = image_size)
val_gen = DataGen(path_input =  "content/val2014", path_mask =  "content/mask_val_2014", batch_size = batch_size , image_size = image_size)


train_steps =  len(os.listdir( "content/train2014"))/batch_size
logger = tf.keras.callbacks.TensorBoard(log_dir=log_dir,  update_freq='epoch', profile_batch=0,  histogram_freq=1)

save_model_batch = SaveModelEachBatch('models23')
model_check = ModelCheckpoint('models_fdgfdf/weights.{epoch:02d}-{loss:.5f}.hdf5', monitor='loss', verbose=0, save_best_only=True)

model.fit_generator(train_gen , validation_data = val_gen , 
                    steps_per_epoch = train_steps , epochs=epochs,
                    callbacks=[model_check,save_model_batch, logger])

# custom_version(model, '2666793940.jpg')