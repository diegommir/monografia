import os
import pandas
import shutil
import pydicom

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
    not_found = 0

def copy_two(row):
    '''
    This function is used to copy other files (ROI and mask) 
    to the new database directory.
    '''
    #If it is not a row with two images, stop
    if row['num_img'] != 2:
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

    #Count total images
    counter.i += 2
    #Count files that were not found
    if not os.path.exists(src_roi):
        counter.not_found += 1
    if not os.path.exists(src_mask):
        counter.not_found += 1

    #If file wasn't copied yet...
    if not os.path.exists(dest_roi):
        print('Copying {}...'.format(file_name_roi))
        shutil.copyfile(src_roi, dest_roi)

    #If there is too images within directory and the file wasn't copied yet...
    if not os.path.exists(dest_mask):
        print('Copying {}...'.format(file_name_mask))
        shutil.copyfile(src_mask, dest_mask)
    
    print('Counter: {}'.format(counter.i))

def correct_roimask(row):
    '''
    This function is used to verify if the files ROI and Mask were copied
    correctly to their respectively directories. If not, apply some correction.
    This function works in conjunction with the 'copy_two' function. I did it 
    separately because I only realized some files were not following the pattern 
    after checking the results of 'copy_two'. And once it take some time to run,
    it is faster to just check and correct files afterwards.
    '''
    #If it is not a row with two images, stop
    if row['num_img'] != 2:
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
    dest_roi = path_roi + '/' + file_name_roi
    dest_mask = path_mask + '/' + file_name_mask

    img_ds_roi = pydicom.dcmread(dest_roi)
    img_ds_mask = pydicom.dcmread(dest_mask)
    img_arr_roi = img_ds_roi.pixel_array
    img_arr_mask = img_ds_mask.pixel_array

    #If the pixel count of the ROI image is bigger than pixel count of Mask image,
    #then they need to swap places
    if (len(img_arr_roi) * len(img_arr_roi[0])) > (len(img_arr_mask) * len(img_arr_mask[0])):
        print('Swapping: {} and {}'.format(file_name_roi, file_name_mask))
        shutil.move(dest_roi, dest_roi + '.temp')
        shutil.move(dest_mask, dest_roi)
        shutil.move(dest_roi + '.temp', dest_mask)
    
    #Count total images
    counter.i += 2
    print('Counter: {}'.format(counter.i))

#df.apply(copy_full, axis=1)
#df.apply(copy_two, axis=1)
#df.apply(correct_roimask, axis=1)

print('Total images: ', counter.i)
print('Not found: ', counter.not_found)

print(df['num_img'].sum())

print(df[
    (df['seq'] == 0)
]['num_img'].sum())

print(df[
    (df['seq'] > 0) &
    (df['num_img'] == 2)
]['num_img'].sum())


print(
    df[
        (df['id'] == 'P_02176')
    ].sort_values('seq')
)