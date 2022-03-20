import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import matplotlib
import glob
import cv2

def save_image(filename, i = 1):
    img_array = np.load(filename)
    print((img_array.shape))
    # img_array = img_array[0,...]
    img_array = img_array[0, :, :, i]
    print((img_array.shape))
    plt.imshow(img_array, cmap="gray")
    img_name = filename + ".png"
    matplotlib.image.imsave(img_name, img_array)
    print(img_name)
    return img_name


def work_with_image_test(original, filename):
    from keras.metrics import MeanIoU
    original_array = mpimg.imread(original)
    img_array = np.load(filename)    
    print(("Original Shape ",  original_array.shape))
    print(("Segmented Shape ",  img_array.shape))
    image_count, width,height, segments = img_array.shape
    y_max = np.argmax(img_array, axis=3)
    y_max_image = np.argmax(img_array, axis=3)[0, :, :]
    print(y_max_image)
    filename1 = 'savedImage-' +  filename +'.png'
    #print("shape, " y_max_image.shape)
    # Using cv2.imwrite() method
    # Saving the image
    h = np.bincount(y_max_image.ravel())
    print (h)

    #cv2.imwrite(filename1, y_max_image)
  #  IoU_keras = MeanIoU(num_classes=segments)
  #  IoU_keras.update_state(original_array, y_max_image)
  #  print("Mean IoU ", IoU_keras.result())
    plt.figure(figsize=(12,8))
    plt.subplot(231)
    plt.title("original")
    plt.imshow(original_array, cmap='jet')
    plt.subplot(232)
    plt.title("predict img")
    plt.imshow(y_max_image, cmap='jet')
    plt.savefig("result.png")

#save_image("x1_test.npy")
#save_image("y1_pred.npy")
#for  i in range(3,50):
#    save_image("y1_pred.npy", i)
#    t = input()
work_with_image_test(original="../../images/train2017/000000000009.jpg", filename='y1_pred.npy')
