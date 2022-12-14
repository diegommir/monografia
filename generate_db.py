import os
import pandas
import shutil

df = pandas.read_csv('new_meta.csv')

print(df)
print(df['num_img'].sum())

types = ['full mammogram images', 'ROI mask images', 'cropped images']

base_dir = '/Volumes/Seagate/monografia/database'

def copy_full(row):
    #If not a full mammogram, stop
    if row['seq'] > 0:
        return
    
    path = base_dir
    if row['db'] == 'Test':
        path += '/full/test'
    else:
        path += '/full/train'
    
    file_name = '{}_{}_{}_{}_{}.dcm'.format(row['id'], row['seq'], row['type'], row['side'], row['view'])

    src = base_dir + '/' + row['File Location'] + '/1-1.dcm'
    dest = path + '/' + file_name

    print('Copying {}...'.format(file_name))
    shutil.copyfile(src, dest)

def copy_others(row):
    #If it is a full mammogram, stop
    if row['seq'] == 0:
        return
    
    path_roi = base_dir
    path_mask = base_dir
    if row['db'] == 'Test':
        path_roi += '/roi/test'
        path_mask += '/mask/test'
    else:
        path_roi += '/roi/train'
        path_mask += '/mask/train'
    
    file_name_roi = '{}_{}_{}_{}_{}_roi.dcm'.format(row['id'], row['seq'], row['type'], row['side'], row['view'])
    file_name_mask = '{}_{}_{}_{}_{}_mask.dcm'.format(row['id'], row['seq'], row['type'], row['side'], row['view'])

    src_roi = base_dir + '/' + row['File Location'] + '/1-1.dcm'
    src_mask = base_dir + '/' + row['File Location'] + '/1-2.dcm'
    dest_roi = path_roi + '/' + file_name_roi
    dest_mask = path_mask + '/' + file_name_mask

    print('Copying {}...'.format(file_name_roi))
    shutil.copyfile(src_roi, dest_roi)
    print('Copying {}...'.format(file_name_mask))
    shutil.copyfile(src_mask, dest_mask)

#df.apply(copy_full, axis=1)
df.apply(copy_others, axis=1)

print(df[
    (df['seq'] != 0)
].count())