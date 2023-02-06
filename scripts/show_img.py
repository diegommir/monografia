import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from torchvision import transforms

path = '/full/train/P_00034_0_Calc_RIGHT_CC.dcm'

base_dir = '/Volumes/Seagate/monografia/database'

img_ds = pydicom.dcmread(base_dir + path)
img_arr = img_ds.pixel_array
img_arr = np.uint8(img_arr / 256)
image = Image.fromarray(img_arr)
t = transforms.ToTensor()(image)
print(image)
print(t.size())
print(img_arr)
print(t)

image.save('./image.png')

top = 100
bottom = 200
left = 100
right = 200

rect = Rectangle((top, left), right - left, bottom - top, linewidth=1, edgecolor='r')

#plt.imshow(image, cmap='gray')
#plt.gca().add_patch(rect)
#plt.show()
