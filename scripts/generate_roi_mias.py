import os
import pandas as pd
from random import random
from PIL import Image, ImageOps

class NewCsv:
    data = []
    i = 0

def generate_roi(row):
        image_size = 256
        x = row['x']
        y = row['y']
        clazz = 1
        severity = 1 if row['severity'] == 'M' else 0

        #Using mean values
        if row[2] == 'NORM':
            clazz = 0
            x = 488
            y = 520
        
        db = 0
        #Separate train test
        if random() >= 0.2:
            db = 1

        file_name = row['ref']
        file_path = os.path.join(db_dir, '{}.pgm'.format(file_name))

        print(file_name)
        img = Image.open(file_path)

        #Define crop bounds
        left = int(x) - int(image_size / 2)
        top =  img.height - int(y) - int(image_size / 2)
        right = int(x) + int(image_size / 2)
        bottom = img.height - int(y) + int(image_size / 2)

        #Crop ROI
        img = img.crop((left, top, right, bottom))
        #Save original ROI
        img.save(os.path.join(roi_dir, '{}_0.bmp'.format(file_name)))
        new_ref = '{}_0'.format(file_name)
        NewCsv.data.append([new_ref, clazz, severity, db])

        #Augmentation
        img_aug = img.rotate(90)
        img_aug.save(os.path.join(roi_dir, '{}_1.bmp'.format(file_name)))
        new_ref = '{}_1'.format(file_name)
        NewCsv.data.append([new_ref, clazz, severity, db])

        img_aug = img.rotate(180)
        img_aug.save(os.path.join(roi_dir, '{}_2.bmp'.format(file_name)))
        new_ref = '{}_2'.format(file_name)
        NewCsv.data.append([new_ref, clazz, severity, db])

        img_aug = img.rotate(270)
        img_aug.save(os.path.join(roi_dir, '{}_3.bmp'.format(file_name)))
        new_ref = '{}_3'.format(file_name)
        NewCsv.data.append([new_ref, clazz, severity, db])

        img_aug = ImageOps.flip(img)
        img_aug.save(os.path.join(roi_dir, '{}_4.bmp'.format(file_name)))
        new_ref = '{}_4'.format(file_name)
        NewCsv.data.append([new_ref, clazz, severity, db])

        img_aug = ImageOps.mirror(img)
        img_aug.save(os.path.join(roi_dir, '{}_5.bmp'.format(file_name)))
        new_ref = '{}_5'.format(file_name)
        NewCsv.data.append([new_ref, clazz, severity, db])

        NewCsv.i += 1

db_dir = '/Volumes/Seagate/monografia/all-mias'
roi_dir = '/Volumes/Seagate/monografia/all-mias/roi'

# Loading metadata
cwd = os.getcwd()
df = pd.read_csv(os.path.join(cwd, 'csv/mias.csv'))

df = df.drop(df[
    (df['severity'] == 'B') &
    (pd.isna(df['radius']))
].index)
df = df.drop(df[
    (df['severity'] == 'M') &
    (pd.isna(df['radius']))
].index)

df['x'] = pd.to_numeric(df['x'], errors='coerce')
df['y'] = pd.to_numeric(df['y'], errors='coerce')
df['radius'] = pd.to_numeric(df['radius'], errors='coerce')
print(df.dtypes)

print(df.head())
print(df.count())
print(df.describe())

print(df[
    (df['ref'] == 'mdb025')
])

df.apply(generate_roi, axis=1)

column_names = ['ref', 'class', 'severity', 'db']
df = pd.DataFrame(NewCsv.data, columns=column_names)
df.to_csv(os.path.join(cwd, 'csv/new_mias.csv'), index=False)

print(NewCsv.i)
print(len(NewCsv.data))
