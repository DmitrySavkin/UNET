import os
import tensorflow as tf
from time import time
from keras.callbacks import TensorBoard, Callback
from telepyth import TelepythClient
from datetime import datetime
#import tensorflow.compat.v1 as tf


class TensorBoardBatchLogger(TensorBoard):
    def __init__(self, project_path, batch_size, log_every=1, VERBOSE=0, **kwargs):
        pass