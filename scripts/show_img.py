import os
import pydicom
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

path = '/roi/train/P_01735_1_Mass_RIGHT_CC_roi.dcm'

base_dir = '/Volumes/Seagate/monografia/database'

img_ds = pydicom.dcmread(base_dir + path)
img_arr = img_ds.pixel_array

top = 100
bottom = 200
left = 100
right = 200

rect = Rectangle((top, left), right - left, bottom - top, linewidth=1, edgecolor='r')

plt.imshow(img_arr, cmap='gray')
#plt.gca().add_patch(rect)
plt.show()
