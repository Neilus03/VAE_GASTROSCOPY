#In this file we will transform and store the images in a new folder as .bmp files.

import os
from PIL import Image

# Path to the folder containing the images
path_png_images = "/home/ndelafuente/CVC/EGD_Barcelona/gastroscopy_attention_classifier/TRAIN&MODEL_4_GENERATEDDATA/data/train/2"

# Path to the folder where the .bmp images will be stored
path_bmp_images = "/home/ndelafuente/CVC/EGD_Barcelona/gastroscopy_attention_classifier/TRAIN&MODEL_4_GENERATEDDATA/data/trainbmp/2"
#ifr the folder does not exist, create it
if not os.path.exists(path_bmp_images):
    os.makedirs(path_bmp_images)

for i, filename in enumerate(os.listdir(path_png_images)):
    if filename.endswith(".png"):
        im = Image.open(path_png_images + "/" + filename)
        im.save(path_bmp_images + "/" + str(int(i)+200) + ".bmp")
        continue
    else:
        continue