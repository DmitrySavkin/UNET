import os
import sys
from keras import Input, Model
from keras.callbacks import ModelCheckpoint,  TensorBoard, Callback
from another_unet import CreateModel
# from utils_folder.logger import SaveModelEachBatch
from unet import get_unet, build_model
from metrics import iou_loss_core
from generator import KerasGenerator
from utils_folder import config
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_PATH) # чтобы из консольки можно было запускать




def fix_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)


# fix_gpu()

tf.debugging.set_log_device_placement(True)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))



config = config.CocoConfig()


keras_gen = KerasGenerator(annFile='../../coco_dataset/annotations/instances_train2017.json',
                     dataset_dir='coco_dataset',
                     subset='train',
                     year='2017',
                     batch_size=2)
gen = keras_gen.generate_batch()

img_size_target = (config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM, 3)

# input_img = Input(img_size_target, name='img')
# model = get_unet(input_img, exit_channels=keras_gen.num_cats, n_filters=8, dropout=0.05, batchnorm=True)
# model.summary()

# Netz 2
model_ = CreateModel(img_size_target=config.IMAGE_MAX_DIM)
model_.compile(loss='binary_crossentropy', optimizer="adam", metrics=[iou_loss_core])
model_.summary()

# Learning:
model_check = ModelCheckpoint('../../models/weights.{epoch:02d}-{loss:.5f}.hdf5', monitor='loss', verbose=2, save_best_only=True)
tf_logger = TensorBoard(log_dir='./logs', histogram_freq=0,
                          write_graph=True, write_images=False) #  = TensorBoardBatchLogger(project_path=PROJECT_PATH, batch_size=keras_gen.batch_size)
#save_model_batch = SaveModelEachBatch('../../models')
history = model_.fit_generator(gen,
                              steps_per_epoch=100, #keras_gen.total_imgs // keras_gen.batch_size,
                              epochs=3,
			      
                              callbacks=[tf_logger, model_check])
