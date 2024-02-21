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
import argparse

class AVEDataset(Dataset):

    def __init__(self, args, mode='train'):
        self.args = args
        self.image = []
        self.audio = []
        self.label = []
        self.mode = mode
        classes = []

        self.data_root = self.args.data_path
        # class_dict = {'NEU':0, 'HAP':1, 'SAD':2, 'FEA':3, 'DIS':4, 'ANG':5}

        self.visual_feature_path = self.data_root
        self.audio_feature_path = os.path.join(self.data_root, 'Audio-1004-SE') 

        self.train_txt = os.path.join(self.data_root, 'trainSet.txt')
        self.test_txt = os.path.join(self.data_root, 'testSet.txt')
        self.val_txt = os.path.join(self.data_root, 'valSet.txt')

        if mode == 'train':
            txt_file = self.train_txt
        elif mode == 'test':
            txt_file = self.test_txt
        else:
            txt_file = self.val_txt

        with open(self.test_txt, 'r') as f1:
            files = f1.readlines()
            for item in files:
                item = item.split('&')
                if item[0] not in classes:
                    classes.append(item[0])
        class_dict = {}
        for i, c in enumerate(classes):
            class_dict[c] = i

        with open(txt_file, 'r') as f2:
            files = f2.readlines()
            for item in files:
                item = item.split('&')
                audio_path = os.path.join(self.audio_feature_path, item[1] + '.pkl')
                visual_path = os.path.join(self.visual_feature_path, 'Image-{:02d}-FPS-SE'.format(1), item[1])

                if os.path.exists(audio_path) and os.path.exists(visual_path):
                    if audio_path not in self.audio:
                        self.image.append(visual_path)
                        self.audio.append(audio_path)
                        self.label.append(class_dict[item[0]])
                else:
                    continue


    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):

        spectrogram = pickle.load(open(self.audio[idx], 'rb'))

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
        image_samples = os.listdir(self.image[idx])
        # select_index = np.random.choice(len(image_samples), size=self.args.num_frame, replace=False)
        # select_index.sort()

        num_frame = 4 # according to PMR paper: https://arxiv.org/pdf/2211.07089.pdf
        images = torch.zeros((num_frame, 3, 224, 224))
        for i in range(num_frame):
            # for i, n in enumerate(select_index):
            img = Image.open(os.path.join(self.image[idx], image_samples[i])).convert('RGB')
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

    train_set = AVEDataset(args, mode='train')
    val_set = AVEDataset(args, mode='val')
    test_set = AVEDataset(args, mode='test')

    return (train_set, val_set, test_set)

if __name__ == "__main__": 
    dirpath = "../data/ave"
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