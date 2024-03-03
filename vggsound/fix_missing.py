import tarfile
import os
import pandas as pd

data_root = "../data/vggsound"

# set of existing files
train_files = os.listdir(os.path.join(data_root, 'train'))
test_files = os.listdir(os.path.join(data_root, 'test'))
existing_files = set(train_files).union(set(test_files))

# The path to your CSV file
csv_path = 'vggsound.csv'

# Read the CSV into a DataFrame and create file to folder mapping
df = pd.read_csv(csv_path, header=None)
df['filename'] = df[0] + "_" + df[1].apply(lambda x: str(x).zfill(6)) + '.mp4'
file_names = set(list(df['filename']))

missing_files = sorted(list(file_names - existing_files))

filtered_df = df[~df['filename'].isin(missing_files)]
filtered_df = filtered_df.drop('filename', axis=1)

filtered_df.to_csv('vggsound_corrected.csv', index=False, header=False)
print("Done")