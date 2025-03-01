import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import rasterio
from skimage.transform import resize

def greyscalify(image_dir: str) -> Image:

  lidar = rasterio.open(image_dir)
  lidar_elevation = lidar.read(1)

  elevation_max = np.max(lidar_elevation)
  elevation_min = np.min(lidar_elevation)

  normalised_lidar = (lidar_elevation - elevation_min)/(elevation_max - elevation_min)*255

  global lidar_image

  lidar_image = Image.fromarray(normalised_lidar)

  return lidar_image

#test
lidar_grey = greyscalify('data/DSM_TQ0075_P_12757_20230109_20230315.tif')
lidar_grey.show()
print(lidar_grey.size)

def dimensions(path):
   
    image = Image.open(path)
    width, height = image.size
    if (width/height) == 1:
        image = image.thumbnail((lidar_image.size))
    else:
        image = image.resize((lidar_image.size), Image.LANCZOS)
    return image 




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



image1 = dimensions('data/satellite.jpg')
image1.load()

data1 = np.asarray(image1, dtype="int32")


data2 = np.asarray(lidar_grey, dtype="int32")
lidar_resize = resize(data2, (500, 500))
print(lidar_resize.shape)

panimage = pansharpen(data1, data2)
panimage.show()