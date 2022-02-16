from keras.models import load_model
import numpy as np
from generator import KerasGenerator
from metrics import iou_loss_core


def predict(path='../../models/weights.04-0.01406.hdf5'):
    model = load_model(path, custom_objects={'iou_loss_core': iou_loss_core})

    keras_gen = KerasGenerator(annFile='../../coco_dataset/annotations/instances_val2017.json',
                        dataset_dir='coco_dataset',
                        subset='val',
                        year='2017',
                        batch_size=2)
    gen = keras_gen.generate_batch()
    x_train, y_train = next(gen)


    y_pred = np.array(model.predict(x_train))
    image_id = np.array(keras_gen.batch_train_indecies[0])

    np.save('y_pred.npy', y_pred)
    np.save('x_test.npy', x_train)
    np.save('image_id.npy', image_id)
 



if __name__ == '__main__':
    predict()
