# VGGSound Dataset Download and Preprocessing

1. Download dataset from HuggingFace. I recommend creating a screen instance so it can be done in the background. Make sure you have git lfs installed. Install git lfs on ubuntu (sudo apt-get install git-lfs). Make sure you clone it in the data folder. This may take a few hours since the huggingface network bandwidth is about 30 MB/s and the dataset is about 338GB, so it will take a bit more than 3 hours to download the entire dataset. Note that there is no progress bar, but you can monitor if things are being downloaded if you run 'du' on the folder and see the number of bytes go up. 
```bash 
screen -S vggsound
git clone https://huggingface.co/datasets/Loie/VGGSound
```

2. The 'vggsound_08.tar.gz' file is corrupted. I've manually extracted the files I could but lost about 2GB/1219 vids/0.612% of the total data. You can download the new version [here] and move it into the data folder. 

2. Rename the folder from VGGSound to vggsound and create train and test folders within the vggsound folder. 

TODO: correct the vggsound.csv file
3. Copy the train/test split csv file by creating a new file and copying the contents of the original file. 
```bash 
touch vggsound.csv
```
Now just ctrl+c the contents into the file from [source](https://raw.githubusercontent.com/GeWu-Lab/OGM-GE_CVPR2022/main/data/VGGSound/vggsound.csv)

4. Run `make_train_test_split.py`, make sure you change the data_root in the file. This script will create the train and test folders and move the files to the appropriate folders. Note


Todo: 
- After file extraction, write a script to remove entries from vggsound.csv that don't exist in the dataset cuz there's been some data loss.
- upload my version of vggsound_08.tar.gz to huggingface git lfs
- Create label mappings based on csv file 