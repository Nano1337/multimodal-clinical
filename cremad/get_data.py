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
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
import torchaudio.transforms as T
import torchaudio
import argparse

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


class CremadDataset(Dataset):

    def __init__(self, args, mode='train'):
        self.args = args
        self.image = []
        self.audio = []
        self.label = []
        self.mode = mode

        self.data_root = self.args.data_path
        class_dict = {'NEU':0, 'HAP':1, 'SAD':2, 'FEA':3, 'DIS':4, 'ANG':5}

        self.visual_feature_path = self.data_root
        self.audio_feature_path = os.path.join(self.data_root, 'Audio-1004') 

        self.train_csv = os.path.join(self.data_root, 'train.csv')
        self.test_csv = os.path.join(self.data_root, 'test.csv')

        if mode == 'train':
            csv_file = self.train_csv
        else:
            csv_file = self.test_csv

        with open(csv_file, encoding='UTF-8-sig') as f2:
            csv_reader = csv.reader(f2)
            for item in csv_reader:
                audio_path = os.path.join(self.audio_feature_path, item[0] + '.pkl')
                visual_path = os.path.join(self.visual_feature_path, 'Image-{:02d}-FPS'.format(1), item[0])

                if os.path.exists(audio_path) and os.path.exists(visual_path):
                    self.image.append(visual_path)
                    self.audio.append(audio_path)
                    self.label.append(class_dict[item[1]])
                else:
                    continue

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):

        spectrogram = pickle.load(open(self.audio[idx], 'rb'))

        if self.mode == 'train':
            visual_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                # transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
                # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                # transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False), # operates on tensor
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            visual_transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # Visual
        image_samples = os.listdir(self.image[idx])
        # select_index = np.random.choice(len(image_samples), size=self.args.num_frame, replace=False)
        # select_index.sort()

        # NOTE: num_frames setting here
        num_frame = 3 # according to PMR paper: https://arxiv.org/pdf/2211.07089.pdf


        images = torch.zeros((num_frame, 3, 224, 224))
        for i in range(num_frame):
            # for i, n in enumerate(select_index):
            img = Image.open(os.path.join(self.image[idx], image_samples[i])).convert('RGB')
            img = visual_transform(img)
            images[i] = img

        images = torch.permute(images, (1,0,2,3))

        # Audio
        if self.mode == 'train':
            spectrogram = apply_spec_augment(
                spectrogram, 
                freq_mask_param=30,  
                time_mask_param=120, 
                num_freq_masks=2,  
                num_time_masks=3    
            )

        # label
        label = self.label[idx]

        if self.args.model_type == 'qmf':
            return spectrogram, images, label, idx
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

    train_set = CremadDataset(args, mode='train')
    test_set = CremadDataset(args, mode='test')
    val_set = test_set

    return (train_set, val_set, test_set)

if __name__ == "__main__": 
    dirpath = "../data/cremad"
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    setattr(args, 'data_path', dirpath)

    train_set, val_set, test_set = get_data(args)

    train_sampler = make_balanced_sampler(train_set.label)

    train_loader = DataLoader(
        train_set, 
        batch_size=16, 
        collate_fn=train_set.custom_collate, 
        sampler=train_sampler
    )

    batch = next(iter(train_loader))
    print(f'x1: {batch[0].shape}, x2: {batch[1].shape}, label: {batch[2].shape}')
    print(batch[2])