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

  normalised_lidar = (lidar_elevation - elevation_min)/(elevation_max - elevation_min)

  global lidar_image

  lidar_image = Image.fromarray(normalised_lidar)

  return lidar_image

#test



"""def dimensions(image: Image):
   
    image = image.resize((5000, 5000), Image.LANCZOS)
    return image 
"""



def pansharpen(image_array1, image_array2):
    rgb_values = np.array([0.299, 0.587, 0.114])
    psuedo_pan_array = np.true_divide((image_array1*rgb_values).sum(axis=2), rgb_values.sum())
    ratio = image_array2/psuedo_pan_array
    new_red = image_array1[:, :, 0] * ratio
    new_green = image_array1[:, :, 1] * ratio
    new_blue = image_array1[:, :, 2] * ratio
    pansharpened_array = np.stack([new_red, new_green, new_blue], axis=2)
    pansharpened_image = Image.fromarray(pansharpened_array.astype(np.uint8))
    return pansharpened_image



image1 = Image.open('data/satellite.jpg')
image1 = image1.resize((5000, 5000), Image.LANCZOS)
image1.show()
print(image1.size)
data1 = np.asarray(image1, dtype="int32")




lidar_grey = greyscalify('data/DSM_TQ0075_P_12757_20230109_20230315.tif')
lidar_grey= lidar_grey.resize((5000, 5000), Image.LANCZOS)
data2 = np.asarray(lidar_grey, dtype="int32")


panimage = pansharpen(data1, data2)
panimage.show()