import os
from tqdm import tqdm
# train_videos = '/data/users/xiaokang_peng/VGGsound/train-videos/train_video_list.txt'
# test_videos = '/data/users/xiaokang_peng/VGGsound/test-videos/test_video_list.txt'

# train_audio_dir = '/data/users/xiaokang_peng/VGGsound/train-audios/train-set'
# test_audio_dir = '/data/users/xiaokang_peng/VGGsound/test-audios/test-set'

train_audio_dir = '../data/vggsound/audio/train'
test_audio_dir = '../data/vggsound/audio/test'

cvs_path = 'vggsound_corrected.csv'
data_root = '../data/vggsound'

# create directories if they don't exist
if not os.path.exists(train_audio_dir):
    os.makedirs(train_audio_dir)
if not os.path.exists(test_audio_dir):
    os.makedirs(test_audio_dir)

train_files = os.listdir(os.path.join(data_root, 'train'))
test_files = os.listdir(os.path.join(data_root, 'test'))

# test set processing
print("Processing test files")
for vid_file in tqdm(test_files):
    mp4_filename = os.path.join(data_root, 'test', vid_file)
    wav_filename = os.path.join(test_audio_dir, vid_file[:-4]+'.wav')
    if os.path.exists(wav_filename):
        pass
    else:
        os.system('ffmpeg -i {} -acodec pcm_s16le -ar 16000 {} > /dev/null 2>&1'.format(mp4_filename, wav_filename))


# train set processing
print("Processing train files")
for vid_file in tqdm(train_files):
    mp4_filename = os.path.join(data_root, 'train', vid_file)
    wav_filename = os.path.join(train_audio_dir, vid_file[:-4]+'.wav')
    if os.path.exists(wav_filename):
        pass
    else:
        os.system('ffmpeg -i {} -acodec pcm_s16le -ar 16000 {} > /dev/null 2>&1'.format(mp4_filename, wav_filename))



