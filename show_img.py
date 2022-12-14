import os
import pydicom
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

#/Volumes/Seagate/monografia/database/CBIS-DDSM/Calc-Training_P_01437_LEFT_CC/08-07-2016-DDSM-NA-06269/1.000000-full mammogram images-26717/1-1.dcm
#/Volumes/Seagate/monografia/database/CBIS-DDSM/Calc-Training_P_01437_LEFT_CC_1/09-06-2017-DDSM-NA-30776/1.000000-ROI mask images-59186/1-1.dcm
#/Volumes/Seagate/monografia/database/CBIS-DDSM/Calc-Training_P_01437_LEFT_CC_2/09-06-2017-DDSM-NA-95450/1.000000-ROI mask images-78625/1-1.dcm
#/Volumes/Seagate/monografia/database/CBIS-DDSM/Calc-Training_P_01437_LEFT_CC_3/09-06-2017-DDSM-NA-40188/1.000000-ROI mask images-71745/1-1.dcm

path = '/Calc-Training_P_01437_LEFT_CC/08-07-2016-DDSM-NA-06269/1.000000-full mammogram images-26717/1-1.dcm'
path = '/Calc-Training_P_01437_LEFT_CC_1/09-06-2017-DDSM-NA-30776/1.000000-ROI mask images-59186/1-1.dcm'
path = '/Calc-Training_P_01437_LEFT_CC_2/09-06-2017-DDSM-NA-95450/1.000000-ROI mask images-78625/1-1.dcm'
path = '/Calc-Training_P_01437_LEFT_CC_3/09-06-2017-DDSM-NA-40188/1.000000-ROI mask images-71745/1-1.dcm'

path = '/Calc-Training_P_01437_LEFT_CC_3/09-06-2017-DDSM-NA-40188/1.000000-ROI mask images-71745/1-1.dcm'

base_dir = '/Volumes/Seagate/monografia/database/CBIS-DDSM'

img_ds = pydicom.dcmread(base_dir + path)
img_arr = img_ds.pixel_array

print(img_arr.shape)

top = 100
bottom = 200
left = 100
right = 200

rect = Rectangle((top, left), right - left, bottom - top, linewidth=1, edgecolor='r')

plt.imshow(img_arr, cmap='gray')
plt.gca().add_patch(rect)
plt.show()
