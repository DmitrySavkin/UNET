import os
import sys
import tensorflow as tf
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
batch_size = 3
# Генератор данных:
keras_gen = KerasGenerator(annFile='../../coco_dataset/annotations/instances_train2017.json',
                     dataset_dir='coco_dataset',
                     subset='train',
                     year='2017',
                     batch_size=batch_size)
train_gen = keras_gen.generate_batch()
step = keras_gen.total_imgs // batch_size
# Генератор данных:
keras_gen = KerasGenerator(annFile='../../coco_dataset/annotations/instances_val2017.json',
                     dataset_dir='coco_dataset',
                     subset='val',
                     year='2017',
                     batch_size=batch_size)
val_gen = keras_gen.generate_batch()

img_size_target = (128, 128, 3)
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
tf_logger = tf.keras.callbacks.TensorBoard(log_dir=log_dir,  update_freq='epoch', profile_batch=0,  histogram_freq=1)
model_check = ModelCheckpoint('../../models/weights.{epoch:02d}-{loss:.5f}.hdf5', monitor='loss', verbose=0, save_best_only=True)
# tf_logger = TensorBoardBatchLogger(project_path=PROJECT_PATH, batch_size=keras_gen.batch_size)
save_model_batch = SaveModelEachBatch('../../models')
history = model_.fit(train_gen, 
                              validation_data=val_gen , 
                              steps_per_epoch=step,
                              epochs=30,
                              callbacks=[ model_check, tf_logger, save_model_batch])
print('after')