import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        input("Ready")
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

import os
import sys
from keras import Input, Model
from keras.callbacks import ModelCheckpoint

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_PATH) # чтобы из консольки можно было запускать

from another_unet import CreateModel
from utils_folder.logger import TensorBoardBatchLogger, SaveModelEachBatch
from unet import get_unet, build_model
from metrics import iou_loss_core
from generator import KerasGenerator
from utils_folder import config

config = config.CocoConfig()

# Генератор данных:
keras_gen = KerasGenerator(annFile='coco_dataset/annotations/instances_train2017.json',
                     dataset_dir='coco_dataset',
                     subset='train',
                     year='2017',
                     batch_size=1)
# gen = keras_gen.generate_batch()
keras_gen_1 = KerasGenerator(annFile='coco_dataset/annotations/instances_val2017.json',
                     dataset_dir='coco_dataset',
                     subset='val',
                     year='2017',
                     batch_size=1)
# val_gen = keras_gen_1.generate_batch()
img_size_target = (config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM, 3)

# gen =  MyDataGen(keras_gen)
# val_gen = MyDataGen(keras_gen_1)
# print(gen, val_gen)
# input("Stop")
# Сетка 1:
# input_img = Input(img_size_target, name='img')
# model = get_unet(input_img, exit_channels=keras_gen.num_cats, n_filters=8, dropout=0.05, batchnorm=True)
# model.summary()

# Сетка 2:
model_ = CreateModel(img_size_target=config.IMAGE_MAX_DIM)
model_.compile(loss='binary_crossentropy', optimizer="adam", metrics=[iou_loss_core])
model_.summary()
log_dir = "experiments"
# Обучение:
model_check = ModelCheckpoint('../../models/weights.{epoch:02d}-{loss:.5f}.hdf5', monitor='loss', verbose=0, save_best_only=True)
# tf_logger = TensorBoardBatchLogger(project_path=PROJECT_PATH, batch_size=keras_gen.batch_size)
save_model_batch = SaveModelEachBatch('../../models')
logger = tf.keras.callbacks.TensorBoard(log_dir=log_dir,  update_freq='epoch', profile_batch=0,  histogram_freq=1)
history = model_.fit_generator(keras_gen, validation_data = keras_gen_1,
                              steps_per_epoch=keras_gen.total_imgs // keras_gen.batch_size,
                              epochs=30,
                              callbacks=[ model_check, save_model_batch, logger])