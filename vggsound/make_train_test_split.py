import pandas as pd
import tarfile
import os
import subprocess
import shutil

def decompress_with_fallback(file_path):
    # Primary command with pigz
    command = ['pigz', '-d', '-k', file_path]
    
    try:
        subprocess.run(command, check=True)
        print(f"Decompression of {file_path} completed successfully with pigz.")
    except subprocess.CalledProcessError:
        print(f"pigz failed to decompress {file_path}, attempting fallback to gzip...")
        # Fallback command with gzip
        fallback_command = ['gzip', '-d', '-k', file_path]
        
        try:
            subprocess.run(fallback_command, check=True)
            print(f"Decompression of {file_path} completed successfully with gzip.")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred during decompression with gzip: {e}")


# The path to your .tar.gz file
data_root = '../data/vggsound/'

# The path to your CSV file
csv_path = 'vggsound.csv'

# Read the CSV into a DataFrame and create file to folder mapping
df = pd.read_csv(csv_path, header=None)
df['filename'] = df[0] + "_" + df[1].apply(lambda x: str(x).zfill(6)) + '.mp4'
df['folder'] = df.iloc[:, 3].apply(lambda x: os.path.join(data_root, x))
mapping = pd.Series(df['folder'].values, index=df['filename']).to_dict()

# create train/test folders if they don't exist
os.makedirs(os.path.join(data_root, 'train'), exist_ok=True)
os.makedirs(os.path.join(data_root, 'test'), exist_ok=True)

for i in range(20): 

    tar_gz_path = os.path.join(data_root, 'vggsound_' + str(i).zfill(2) + '.tar.gz')

    decompress_with_fallback(tar_gz_path)

    # Extract files based on the mapping
    with tarfile.open(tar_gz_path[:-3], "r") as tar:
        for member in tar.getmembers():
            if member.name == 'vgg_sound_08_fixed': 
                    continue
            file_name = member.name.split('/')[-1]
            # Get the designated folder ('train' or 'test') from the mapping
            designated_folder = mapping[file_name]
            # Move the file to its designated folder
            tar.extract(member, designated_folder)
            os.rename(os.path.join(designated_folder, member.name), os.path.join(designated_folder, file_name))

    print("Done extracting files from " + tar_gz_path[:-3])
    # clean up 
    if i == 8: 
        shutil.rmtree(os.path.join(data_root, 'train', 'vgg_sound_08_fixed'))
        shutil.rmtree(os.path.join(data_root, 'test', 'vgg_sound_08_fixed'))
    else: 
        shutil.rmtree(os.path.join(data_root, 'train', 'scratch'))
        shutil.rmtree(os.path.join(data_root, 'test', 'scratch'))
    os.remove(tar_gz_path[:-3])
    os.remove(tar_gz_path)

