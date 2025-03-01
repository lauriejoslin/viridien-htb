import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def pansharpen(image_array1, image_array2):
    rgb_values = np.array([0.21, 0.72, 0.07])
    psuedo_pan_array = np.true_divide((image_array1*rgb_values).sum(axis=2), rgb_values.sum())
    ratio = image_array2/psuedo_pan_array
    new_red = image_array1[:, :, 0] * ratio
    new_green = image_array1[:, :, 1] * ratio
    new_blue = image_array1[:, :, 2] * ratio
    pansharpened_array = np.stack([new_red, new_green, new_blue], axis=2)
    pansharpened_image = Image.fromarray(pansharpened_array.astype(np.uint8))
    return pansharpened_image


image1 = Image.open('viridien-htb/data/satellite.jpg')
image1.load()
data1 = np.asarray(image1, dtype="int32")

image2 = Image.open('viridien-htb/data/image_new1.jpg')
print(image2.size)
image2.load()
data2 = np.asarray(image2, dtype="int32")
print(data2.shape)

panimage = pansharpen(data1, data2)
panimage.show()