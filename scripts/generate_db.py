import os
import pandas
import shutil

cwd = os.getcwd()

df = pandas.read_csv(os.path.join(cwd, 'csv/new_meta.csv'))
print(df)

types = ['full mammogram images', 'ROI mask images', 'cropped images']

base_dir = '/Volumes/Seagate/monografia/database'

def copy_full(row):
    '''
    This function is used to copy only the full mammogram images 
    to the new database directory.
    '''
    #If not a full mammogram, stop
    if row['seq'] > 0:
        return
    
    #The files are going to be copied to different directories 
    #depending on if they are from Training or Test databases.
    path = base_dir
    if row['db'] == 'Test':
        path += '/full/test'
    else:
        path += '/full/train'
    
    #Define file name
    file_name = '{}_{}_{}_{}_{}.dcm'.format(row['id'], row['seq'], row['type'], row['side'], row['view'])

    #Define origin and destination paths
    src = base_dir + '/' + row['File Location'] + '/1-1.dcm'
    dest = path + '/' + file_name

    #If file wasn't copied yet...
    if not os.path.exists(dest):
        print('Copying {}...'.format(file_name))
        shutil.copyfile(src, dest)

class counter:
    i = 0

def copy_others(row):
    '''
    This function is used to copy all other files (ROI and mask) 
    to the new database directory.
    '''
    #If it is a full mammogram, stop
    if row['seq'] == 0:
        return
    
    #The files are going to be copied to different directories 
    #depending on if they are from Training or Test databases.
    path_roi = base_dir
    path_mask = base_dir
    if row['db'] == 'Test':
        path_roi += '/roi/test'
        path_mask += '/mask/test'
    else:
        path_roi += '/roi/train'
        path_mask += '/mask/train'
    
    #Define file names
    file_name_roi = '{}_{}_{}_{}_{}_roi.dcm'.format(row['id'], row['seq'], row['type'], row['side'], row['view'])
    file_name_mask = '{}_{}_{}_{}_{}_mask.dcm'.format(row['id'], row['seq'], row['type'], row['side'], row['view'])

    #Define origin and destination paths
    src_roi = base_dir + '/' + row['File Location'] + '/1-1.dcm'
    src_mask = base_dir + '/' + row['File Location'] + '/1-2.dcm'
    dest_roi = path_roi + '/' + file_name_roi
    dest_mask = path_mask + '/' + file_name_mask

    #Count files that were not found
    if not os.path.exists(src_roi):
        counter.i += 1
    if row['num_img'] == 2 and not os.path.exists(src_mask):
        counter.i += 1

    #If file wasn't copied yet...
    if not os.path.exists(dest_roi):
        print('Copying {}...'.format(file_name_roi))
        shutil.copyfile(src_roi, dest_roi)

    #If there is too images within directory and the file wasn't copied yet...
    if row['num_img'] == 2 and not os.path.exists(dest_mask):
        print('Copying {}...'.format(file_name_mask))
        shutil.copyfile(src_mask, dest_mask)

df.apply(copy_full, axis=1)
df.apply(copy_others, axis=1)

print('Not found: ', counter.i)

print(df['num_img'].sum())

print(df[
    (df['seq'] == 0)
]['num_img'].sum())

print(df[
    (df['seq'] != 0)
]['num_img'].sum())
