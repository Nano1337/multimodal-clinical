## Dataset Setup Instructions

1. Use the gdown cli tool to download the AVE_dataset.zip and unzip after. 
```bash
gdown 1FjKwe79e0u96vdjIVwfRQ1V6SoDHe7kK
```

2. Run mp4_to_wav.py to attain audio files from .mp4 files. Ensure paths are correct. 

3. There seems to be something wrong with the Annotation.txt file. Change: 
```
Church bell&VWi2ENBuTbw&good&0&0 -> Church bell&VWi2ENBuTbw&good&0&10
```

4. Run video_preprocessing.py to save frames and spectrograms from mp4 and wav files for the dataloader. Ensure paths are correct

