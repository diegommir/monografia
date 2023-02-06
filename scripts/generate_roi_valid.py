import os
import pandas as pd
from PIL import Image, ImageOps

# Folders
cwd = os.getcwd()
db_dir = os.path.join(cwd, 'valid_db')
roi_dir = os.path.join(cwd, 'valid_db/roi')

class NewCsv:
    data = []

def save_img(img, file_name, i):
    new_ref = '{}_{:02d}'.format(file_name, i)
    img.save(os.path.join(roi_dir, '{}.bmp'.format(new_ref)))
    NewCsv.data.append([new_ref, new_ref.split('_')[0]])

def generate_roi(file):
        img_size = 128

        file_name = file.name.split('.')[0]

        # Open image file
        print('Generating for: ', file_name)
        img = Image.open(os.path.join(db_dir, file))

        # Generate ROI
        img = img.resize((img_size, img_size))
        img = ImageOps.grayscale(img)

        i = 1
        # Augmentation
        for angle in (0, 90, 180, 270):
            # Rotate
            img_aug = img.rotate(angle)
            save_img(img_aug, file_name, i)
            i += 1
            # Mirror
            save_img(ImageOps.mirror(img_aug), file_name, i)
            i += 1

with os.scandir(db_dir) as entries:
    for entry in entries:
        if entry.is_file() and entry.name.endswith('.jpg'):
            generate_roi(entry)

# Save the 
column_names = ['ref', 'class']
df = pd.DataFrame(NewCsv.data, columns=column_names)
df.to_csv(os.path.join(cwd, 'csv/valid.csv'), index=False)

print(df.head())
