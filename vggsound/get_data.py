import copy
import csv
import os
import pickle
import librosa
import numpy as np
from scipy import signal
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler 
from torchvision import transforms
import pdb
import random
import argparse
from torch.utils.data.dataloader import default_collate
import torchaudio

def apply_spec_augment(spectrogram, freq_mask_param=30, time_mask_param=120, num_freq_masks=2, num_time_masks=3):
    """
    Apply SpecAugment (frequency and time masking) to a given spectrogram.
    
    Parameters:
    - spectrogram: Tensor, the input spectrogram of shape [channels, freq, time].
    - freq_mask_param: int, the maximum width of the frequency masks.
    - time_mask_param: int, the maximum length of the time masks.
    - num_freq_masks: int, the number of frequency masks to apply.
    - num_time_masks: int, the number of time masks to apply.

    Returns:
    - Tensor, the augmented spectrogram.
    """
    # Ensure spectrogram is a PyTorch tensor
    if isinstance(spectrogram, np.ndarray):
        spectrogram = torch.from_numpy(spectrogram)

    # Ensure the spectrogram is floating point, as required by torchaudio transforms
    if spectrogram.dtype != torch.float32:
        spectrogram = spectrogram.to(torch.float32)

    # Apply frequency masking
    freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param)
    for _ in range(num_freq_masks):
        spectrogram = freq_mask(spectrogram)

    # Apply time masking
    time_mask = torchaudio.transforms.TimeMasking(time_mask_param)
    for _ in range(num_time_masks):
        spectrogram = time_mask(spectrogram)

    return spectrogram

class VGGSound(Dataset):

    def __init__(self, args, mode='train'):
        self.args = args
        self.mode = mode
        train_video_data = []
        train_audio_data = []
        test_video_data  = []
        test_audio_data  = []
        train_label = []
        test_label  = []
        train_class = []
        test_class  = []

        with open('vggsound_corrected.csv') as f:
            csv_reader = csv.reader(f)

            for item in csv_reader:
                if item[3] == 'train':
                    video_dir = os.path.join(self.args.data_path, 'train_Image-{:02d}-FPS'.format(1), item[0]+'_'+str(item[1]).zfill(6) + ".mp4")
                    audio_dir = os.path.join(self.args.data_path, 'audio/train', item[0]+'_'+str(item[1]).zfill(6)+'.wav')
                    if os.path.exists(video_dir) and os.path.exists(audio_dir) and len(os.listdir(video_dir))>3 :
                        train_video_data.append(video_dir)
                        train_audio_data.append(audio_dir)
                        if item[2] not in train_class: train_class.append(item[2])
                        train_label.append(item[2])

                if item[3] == 'test':
                    video_dir = os.path.join(self.args.data_path, 'test_Image-{:02d}-FPS'.format(1), item[0]+'_'+str(item[1]).zfill(6) + ".mp4")
                    audio_dir = os.path.join(self.args.data_path, 'audio/test', item[0]+'_'+str(item[1]).zfill(6)+'.wav')
                    if os.path.exists(video_dir) and os.path.exists(audio_dir) and len(os.listdir(video_dir))>3:
                        test_video_data.append(video_dir)
                        test_audio_data.append(audio_dir)
                        if item[2] not in test_class: test_class.append(item[2])
                        test_label.append(item[2])

        assert len(train_class) == len(test_class)
        self.classes = train_class

        class_dict = dict(zip(self.classes, range(len(self.classes))))

        if mode == 'train':
            self.video = train_video_data
            self.audio = train_audio_data
            self.label = [class_dict[train_label[idx]] for idx in range(len(train_label))]
        if mode == 'test':
            self.video = test_video_data
            self.audio = test_audio_data
            self.label = [class_dict[test_label[idx]] for idx in range(len(test_label))]


    def __len__(self):
        return len(self.video)

    def __getitem__(self, idx):

        # audio
        sample, rate = librosa.load(self.audio[idx], sr=16000, mono=True)
        while len(sample)/rate < 10.:
            sample = np.tile(sample, 2)

        start_point = random.randint(a=0, b=rate*5)
        new_sample = sample[start_point:start_point+rate*5]
        new_sample[new_sample > 1.] = 1.
        new_sample[new_sample < -1.] = -1.

        spectrogram = librosa.stft(new_sample, n_fft=256, hop_length=128)
        spectrogram = np.log(np.abs(spectrogram) + 1e-7)

        # Audio
        if self.mode == 'train':
            spectrogram = apply_spec_augment(
                spectrogram, 
                freq_mask_param=30, 
                time_mask_param=120, 
                num_freq_masks=2,    
                num_time_masks=3     
            )

        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])


        # Visual
        image_samples = os.listdir(self.video[idx])
        try: 
            select_index = np.random.choice(len(image_samples), size=self.args.use_video_frames, replace=False)
        except: 
            select_index = np.random.choice(len(image_samples), size=self.args.use_video_frames, replace=True) 
        select_index.sort()
        images = torch.zeros((self.args.use_video_frames, 3, 224, 224))
        for i, index in enumerate(select_index):  # Use index from select_index to access image_samples
            img = Image.open(os.path.join(self.video[idx], image_samples[index])).convert('RGB')  
            img = transform(img)
            images[i] = img

        images = torch.permute(images, (1,0,2,3))

        # label
        label = self.label[idx]

        return spectrogram, images, label
    
    def custom_collate(self, batch): 

        batch = default_collate(batch)
        batch[0] = batch[0].unsqueeze(1)

        return batch

def make_balanced_sampler(labels):
    class_counts = torch.bincount(torch.tensor(labels))
    class_weights = 1. / class_counts
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(sample_weights, len(labels), replacement=True)
    return sampler


def get_data(args): 
    train_set = VGGSound(args, mode='train')
    test_set = VGGSound(args, mode='test')
    val_set = test_set

    return train_set, val_set, test_set

if __name__ == "__main__": 
    dirpath = "../data/vggsound"
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    setattr(args, 'data_path', dirpath)
    setattr(args, 'use_video_frames', 3)
    
    train_set, val_set, test_set = get_data(args)

    train_sampler = make_balanced_sampler(train_set.label)

    train_loader = DataLoader(
        train_set, 
        batch_size=16, 
        sampler=train_sampler
    )

    batch = next(iter(train_loader))
    print(f'x1: {batch[0].shape}, x2: {batch[1].shape}, label: {batch[2].shape}')
    print(batch[2])

    batch = next(iter(train_loader))
    print(f'x1: {batch[0].shape}, x2: {batch[1].shape}, label: {batch[2].shape}')
    print(batch[2])

