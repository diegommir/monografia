import os
import pandas as pd
import pydicom

cwd = os.getcwd()
new_meta = pd.read_csv(os.path.join(cwd, 'csv/new_meta.csv'))

#Database path
path_train = '/Volumes/Seagate/monografia/database/roi/train/'
path_test = '/Volumes/Seagate/monografia/database/roi/test/'

def generate_meta_db(path, metadb_name):
    print('Generating {}....'.format(metadb_name))
    metadb = []

    #List all files in the database path
    for file in os.listdir(path):
        #If the file starts with '._', it is a system temp file. Just ignore.
        if file.startswith('._P'):
            continue

        #Try to read the file, otherwise log its name so we can track the problem down.
        img_array = []
        try:
            img_ds = pydicom.dcmread(os.path.join(path, file))
            img_array = img_ds.pixel_array
        except:
            print('Error: {}'.format(file))
            continue

        #Split file name to identify the image
        p, id, seq, typez, side, view, roi = str(file).split('.')[0].split('_')
        id = 'P_' + id

        #Get pathology
        pathology = new_meta[
            (new_meta['id'] == id) &
            (new_meta['seq'] == int(seq)) &
            (new_meta['side'] == side) &
            (new_meta['view'] == view)
        ]['pathology'].iloc[0]

        #Append data to de new database metadata
        metadb.append([id, seq, typez, side, view, len(img_array), len(img_array[0]), pathology, str(file)])

    #Generate a pandas dataframe
    column_names = ['id', 'seq', 'type', 'side', 'view', 'width', 'height', 'pathology', 'file_name']
    metadb = pd.DataFrame(metadb, columns=column_names)
    print(metadb.head())

    #Save dataframe to file
    metadb.to_csv(os.path.join(cwd, 'csv/{}.csv'.format(metadb_name)), index=False)

generate_meta_db(path_train, 'metadb_train')
generate_meta_db(path_test, 'metadb_test')