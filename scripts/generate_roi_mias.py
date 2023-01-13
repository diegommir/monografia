import os
import pandas as pd
from PIL import Image, ImageOps

class NewCsv:
    data = []
    i = 0

def save_img(img, file_name, i, severity, db):
    new_ref = '{}_{:02d}'.format(file_name, i)
    img.save(os.path.join(roi_dir, '{}.bmp'.format(new_ref)))
    NewCsv.data.append([new_ref, severity, db])

def generate_roi(row):
        image_size = 256
        x = row['x']
        y = row['y']
        db = row['db']

        # Set severity
        severity = 'N'
        if row['severity'] == 'M':
            severity = 'M'
        elif row['severity'] == 'B':
            severity = 'B'

        # Using mean values when is a normal image
        if row['class'] == 'NORM':
            x = 488
            y = 520

        # Open image file
        file_name = row['ref']
        file_path = os.path.join(db_dir, '{}.pgm'.format(file_name))
        img = Image.open(file_path)

        new_file_name = '{:03d}_{}'.format(row['id'], file_name)
        print(new_file_name)

        # Define crop bounds
        left = int(x) - int(image_size / 2)
        top =  img.height - int(y) - int(image_size / 2)
        right = int(x) + int(image_size / 2)
        bottom = img.height - int(y) + int(image_size / 2)

        # Crop ROI
        img = img.crop((left, top, right, bottom))
        i = 1

        # Augmentation
        for angle in (0, 90, 180, 270):
            # Rotate
            img_aug = img.rotate(angle)
            save_img(img_aug, new_file_name, i, severity, db)
            i += 1
            # Mirror
            save_img(ImageOps.mirror(img_aug), new_file_name, i, severity, db)
            i += 1

        NewCsv.i += 1

db_dir = '/Volumes/Seagate/monografia/all-mias'
roi_dir = '/Volumes/Seagate/monografia/all-mias/roi'

# Loading metadata
cwd = os.getcwd()
df = pd.read_csv(os.path.join(cwd, 'csv/mias.csv'))

# Drop some incomplete data
df = df.drop(df[
    (df['severity'] == 'B') &
    (pd.isna(df['radius']))
].index)
df = df.drop(df[
    (df['severity'] == 'M') &
    (pd.isna(df['radius']))
].index)

# Force convert this columns to number
df['x'] = pd.to_numeric(df['x'], errors='coerce')
df['y'] = pd.to_numeric(df['y'], errors='coerce')
df['radius'] = pd.to_numeric(df['radius'], errors='coerce')
print(df.dtypes)

# Add an ID column
df['id'] = df.reset_index().index

# 326 Total | 119 lesion | 68 Benignant | 51 Malignant
print(df.head())
print(df.count())
print(df.describe())
print(df[
    df['severity'] == 'B'
]['ref'].count())
print(df[
    df['severity'] == 'M'
]['ref'].count())

# First set everyone as test db
df['db'] = 'test'

# Second separate a train database with balanced classes
df_norm = df[pd.isna(df['severity'])].sample(41, random_state=42)
df_malig = df[df['severity'] == 'M'].sample(41, random_state=42)
df_benig = df[df['severity'] == 'B'].sample(41, random_state=42)

# Now set the separated train db as 'train'
df.loc[df_norm.index, ['db']] = 'train'
df.loc[df_malig.index, ['db']] = 'train'
df.loc[df_benig.index, ['db']] = 'train'

# Verify data
print(df.head())
print(df[
    (df['severity'] == 'B') & 
    (df['db'] == 'train')
]['ref'].count())
print(df[
    (df['severity'] == 'M') & 
    (df['db'] == 'train')
]['ref'].count())
print(df[
    (pd.isna(df['severity'])) & 
    (df['db'] == 'train')
]['ref'].count())

df.apply(generate_roi, axis=1)

# Save the 
column_names = ['ref', 'severity', 'db']
df = pd.DataFrame(NewCsv.data, columns=column_names)
df.to_csv(os.path.join(cwd, 'csv/new_mias.csv'), index=False)

print(NewCsv.i)
print(len(NewCsv.data))
