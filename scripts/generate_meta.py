import os
import pandas

cwd = os.getcwd()

df = pandas.read_csv(os.path.join(cwd, 'csv/metadata.csv'))
df = df[['Subject ID', 'Series Description', 'Number of Images', 'File Location']]

df_calc_train = pandas.read_csv(os.path.join(cwd, 'csv/calc_case_description_train_set.csv'))
df_calc_test = pandas.read_csv(os.path.join(cwd, 'csv/calc_case_description_test_set.csv'))
df_mass_train = pandas.read_csv(os.path.join(cwd, 'csv/mass_case_description_train_set.csv'))
df_mass_test = pandas.read_csv(os.path.join(cwd, 'csv/mass_case_description_test_set.csv'))

df['id'] = df['Subject ID'].apply(lambda id: 'P_' + id.split('_')[2])
df['seq'] = df['Subject ID'].apply(lambda id: int(id.split('_')[5]) if len(id.split('_')) > 5 else 0)
df['type'] = df['Subject ID'].apply(lambda id: id.split('_')[0].split('-')[0])
df['db'] = df['Subject ID'].apply(lambda id: id.split('_')[0].split('-')[1])
df['side'] = df['Subject ID'].apply(lambda id: id.split('_')[3])
df['view'] = df['Subject ID'].apply(lambda id: id.split('_')[4])
df['series'] = df['Series Description']
df['num_img'] = df['Number of Images']

df = df.drop(['Subject ID', 'Series Description', 'Number of Images'], axis=1)

types = ['full mammogram images', 'ROI mask images', 'cropped images']

def get_pathology(row):
    if row['seq'] == 0:
        return pandas.NA

    df_temp = None
    if row['type'] == 'Calc':
        if row['db'] == 'Training':
            df_temp = df_calc_train
        else:
            df_temp = df_calc_test
    else:
        if row['db'] == 'Training':
            df_temp = df_mass_train
        else:
            df_temp = df_mass_test
    
    pathology = df_temp[
        (df_temp['patient_id'] == row['id']) &
        (df_temp['abnormality id'] == row['seq']) &
        (df_temp['left or right breast'] == row['side']) &
        (df_temp['image view'] == row['view'])
    ]['pathology'].iloc[0]

    return pathology

df['pathology'] = df.apply(get_pathology, axis=1)

df.to_csv(os.path.join(cwd, 'csv/new_meta.csv'), index=False)
