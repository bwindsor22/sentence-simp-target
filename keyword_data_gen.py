import pandas as pd
file_path = './full_dataset_all.csv'
data = pd.read_csv(file_path, engine='python')


def remove_bad(df, column_name):
    df = df[df[column_name] != 'pass']
    df = df[df[column_name] != 'Pass']
    df = df.fillna('')
    df = df[df['text'] != '']
    df = df[df[column_name] != '']
    return df

org_norp = data[['text', 'org_norp']]
org_norp_simp = remove_bad(org_norp, 'org_norp')
org_norp_simp['text'] = '<N>' + org_norp_simp['text']
org_norp_simp.columns = ['text', 'target']

org_loc = data[['text', 'org_loc']]
org_loc_simp = remove_bad(org_loc, 'org_loc')
org_loc_simp['text'] = '<L>' + org_loc_simp['text']
org_loc_simp.columns = ['text', 'target']


full_df = pd.concat([org_norp_simp, org_loc_simp])
full_df.to_csv('./single_model.csv')
print('hi')