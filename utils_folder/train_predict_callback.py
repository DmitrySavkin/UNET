import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from . import prediction_util

class PredictionCallBack(keras.callbacks.Callback):  # diff
    """Callback to operate on batch data from metric."""

    def __init__(self, image_names):
        super().__init__()
        self.y_true = None
        self.y_pred = None
        self.image_names = image_names

    def set_model(self, model):
        """Initialize variables when model is set."""
        self.model = model

    def metric(self, y_true, y_pred):
        """Fake metric."""
        self.y_true.assign(y_true)
        self.y_pred.assign(y_pred)

        return 0

    #def on_train_batch_end(self, _batch, _logs=None):
        """See keras.callbacks.Callback.on_train_batch_end."""
    #    prediction_util.prediction(self.model, self.image_name)
    def on_epoch_end(self, epoch, logs=None):
        # output_array = []
        for image_name in self.image_names:
            original_cpy, img_result = prediction_util.prediction(self.model, image_name)
            if output_array is None:
                output_array = np.hstack((original_cpy, img_result)))
            else:
                t = np.hstack((original_cpy, img_result))
                output_array = np.
            cv2.imwrite('result.jpg', np.hstack((original_cpy, img_result)))


    def on_train_end(self, _logs=None):
        """Clean up."""
        del self.y_true, self.y_pred




def tf_nan(dtype):
    """Create NaN variable of proper dtype and variable shape for assign()."""
    return tf.Variable(float("nan"), dtype=dtype, shape=tf.TensorShape(None))