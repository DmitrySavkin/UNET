import cv2
import numpy as np

from keras.models import load_model
from generator import KerasGenerator
from matplotlib import pyplot as plt
from metrics import iou_loss_core
from pycocotools.coco import COCO
from utils_folder.logger import DataCallback
import skimage.io as io


ANNOTATIONS_FILE = '../../content/annotations/instances_val2014.json'
'''
PLEASE USE models23 if nothing works
'''
MODEL_PATH = "../../models23/weights.08-0.30780.hdf5" # '../../models/weights.04-0.01406.hdf5'


 
coco = COCO(ANNOTATIONS_FILE)

def get_class_name(class_id):
    pass 

def predict(model_file, index):
    from keras.metrics import MeanIoU 
    customer_labels = get_customer_labels()
    model = load_model(model_file, custom_objects={'iou_loss_core': iou_loss_core})
    keras_gen = KerasGenerator(annFile='../../content/annotations/instances_val2014.json',
                     dataset_dir='coco_dataset',
                     subset='val',
                     year='2014',
                     batch_size=1)
    gen = keras_gen.generate_batch()
    x_train, y_train = next(gen)
    y_pred = model.predict(x_train)
    print("SHAPE ", y_pred.shape)
    image_id = np.array(keras_gen.batch_train_indecies[0])
    np.save('y_pred.npy', y_pred)
    np.save('x_test.npy', x_train)
    np.save('image_id.npy', image_id)
    y_max_image = np.argmax(y_pred, axis=3)[0, :, :]
    np.save('y_pred.npy', y_pred)
    np.save('x_test.npy', x_train)
    cv2.imwrite(f'x_test_predict_{index}.png', cv2.cvtColor(x_train[0], cv2.COLOR_BGR2RGB))
    h, w = y_max_image.shape 
    img_result = np.zeros((h, w, 3), dtype=np.uint8)
    for  y in range(h):
        for x in range(w):
            id_class = y_max_image[ y, x]
            img_result[y, x , 0] = customer_labels[id_class]['color'][0]
            img_result[y, x , 1] = customer_labels[id_class]['color'][1]
            img_result[y, x , 2] = customer_labels[id_class]['color'][2]

    cv2.imwrite(f'result_predict_{index}.png', img_result)


def custom_version(model_file, image_name, output = 'result.png', image_size = 128):
    customer_labels = get_customer_labels()
    data_callback = DataCallback(path='.')
    print(customer_labels[5])
    model = load_model(model_file, custom_objects={'iou_loss_core': iou_loss_core, 'metric':data_callback.metric})
    original = cv2.imread(image_name, 1)
    original = cv2.resize(original, (image_size , image_size))
    original_cpy = original.copy()
    original = original / 255.0

    print(original.shape)
    #original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
  
    x_train = np.expand_dims(original, 0)
     
    print(x_train.shape, x_train.min(), x_train.max(), x_train.dtype)
    y_pred = model.predict(x_train)
    y_max_image = np.argmax(y_pred, axis=3)[0, :, :]
    print(np.nonzero(y_max_image))
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
    
    
    # print(y_max_image)
    # print(y_max_image.shape, x_train.shape)
    u, count = np.unique(np.array(y_max_image), return_counts=True)
    count_sort_ind = np.argsort(-count)
    print('u[', u[count_sort_ind])
    print('co', count[count_sort_ind])
    for cl in u[count_sort_ind]:
        print(customer_labels[cl]['category'])
    #for cl in np.unique(np.array(y_max_image)):
    #    print(customer_labels[cl]['category'])
        
    #cv2.imwrite(output, img_result)
    cv2.imwrite(output, np.hstack((original_cpy, img_result)))





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
    for keys, val in d.items():
        x = X[randrange(len(X))]
        temp = {"id": keys, "category": val, "color": x}
        result[keys] = temp
        X.remove(x)
    # print(result)
    return result








if __name__ == '__main__':
    import os
    list_1 = ["../../models23/weights.34-0.20345.hdf5",
            "../../models23/weights.33-0.20531.hdf5",
            "../../models23/weights.32-0.20698.hdf5",
            "../../models23/weights.31-0.20807.hdf5",
            "../../models23/weights.30-0.21028.hdf5",
            "../../models23/weights.29-0.21219.hdf5",
            "../../models23/weights.28-0.21376.hdf5",
            "../../models23/weights.27-0.21625.hdf5",
            "../../models23/weights.26-0.21831.hdf5",
            "../../models23/weights.25-0.22032.hdf5",
            "../../models23/weights.24-0.22235.hdf5",
            "../../models23/weights.23-0.22523.hdf5",
            "../../models23/weights.22-0.22735.hdf5"]
    arr = os.listdir("../../models23")
  
  
  
    custom_version(MODEL_PATH, '../../content/train2014/COCO_train2014_000000000061.jpg', output='result_18.03.2022.jpg') 
  
  
  
    i = 0
    for file in []:
        if file.endswith(".hdf5"): 
            print(file,  i)
            custom_version(f"../../models23/{file}", '3.jpg', output = f'result_{file}_.png')
            i += 1
    #pass
    

  