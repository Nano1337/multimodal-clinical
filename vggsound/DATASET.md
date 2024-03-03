# VGGSound Dataset Download and Preprocessing

Complete dataset prep time: 5 hours, not including train set of step 6 

1. Download dataset from HuggingFace. I recommend creating a screen instance so it can be done in the background. Make sure you have git lfs installed. Install git lfs on ubuntu (sudo apt-get install git-lfs). Make sure you clone it in the data folder. This may take a few hours since the huggingface network bandwidth is about 30 MB/s and the dataset is about 338GB, so it will take a bit more than 3 hours to download the entire dataset. Note that there is no progress bar, but you can monitor if things are being downloaded if you run 'du' on the folder and see the number of bytes go up. 
```bash 
screen -S vggsound
git clone https://huggingface.co/datasets/Loie/VGGSound
```

2. The 'vggsound_08.tar.gz' file is corrupted. I've manually extracted the files I could but lost about 2GB/1219 vids/0.612% of the total data. You can download the new version via the script below and move it into the data folder from step 1. 
```bash
git clone https://huggingface.co/datasets/Nano1337/vggsound_08
```

3. Rename the folder from `VGGSound` to `vggsound` and create `train` and `test` folders within the `vggsound` folder. 

4. Run `make_train_test_split.py`, make sure you change the data_root in the file. This script will create the train and test folders and move the files to the appropriate folders. 

5. Run `mp4_to_wav.py` to extract audio spectrograms from the videos. Make sure you modify the paths in the script. This uses ffmpeg and will take about an hour

6. Run `video_preprocessing.py` to extract frames from video so we don't have to decode during dataloading. This is slower than step 5 since we're not using CUDA accelerated libraries for potential compatibility issues (alternatively, modify the code to run FFMpeg's h264_cuvid decoder). Test set frame extraction takes about 30 minutes. Train set frames take much longer. 
