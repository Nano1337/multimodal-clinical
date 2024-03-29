#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from timm.data import create_transform
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import argparse
import os
import json
import numpy as np

class MultimodalFoodDataset(Dataset):
    def __init__(self, args, mode="dev"):
        """
        Args:
            args: Arguments containing data_path.
            mode (string): "train" or "test" to specify the dataset to load.
        """
        data = []
        data2class = {}
        self.args = args
        self.mode = mode
        self.data_path = self.args.data_path
        self.visual_feature_path = os.path.join(self.data_path, "visual", '{}_imgs/'.format(mode))
        self.text_feature_path = os.path.join(self.data_path, "text_token", '{}_token/'.format(mode))
        self.stat_path = os.path.join(self.data_path, "stat_food.txt")
        self.train_txt = os.path.join(self.data_path, "my_train_food.txt")
        self.dev_txt = os.path.join(self.data_path, "my_dev_food.txt")
        self.test_txt = os.path.join(self.data_path, "my_test_food.txt")

        with open(self.stat_path, "r") as f1:
            classes = f1.readlines()
        
        classes = [sclass.strip() for sclass in classes]

        if mode == "train": 
            csv_file = self.train_txt
        elif mode == "dev": 
            csv_file = self.dev_txt
        elif mode == "test": 
            csv_file = self.test_txt
        else: 
            raise NotImplementedError("Please specify one of train/dev/test modes")

        with open(csv_file, "r") as f2:
            csv_reader = f2.readlines()
            for single_line in csv_reader:
                item = single_line.strip().split(".jpg ")
                token_path = os.path.join(self.text_feature_path, item[0] + '_token.npy')
                visual_path = os.path.join(self.visual_feature_path, item[0] + ".jpg.npy")    
                # pdb.set_trace()
                if os.path.exists(token_path) and os.path.exists(visual_path):
                    data.append(item[0])
                    data2class[item[0]] = item[1]
                else:
                    continue

        self.classes = sorted(classes)

        self.data2class = data2class


        if mode == "train":
            mask_path = "/home/haoli/Documents/multimodal-clinical/food101/combined_filter.npy"
            mask = np.load(mask_path)
            if mask is not None:
                if len(mask) != len(data):
                    raise ValueError("Mask length must match the number of samples")
                # Filter data based on the mask
                filtered_data = []
                for i, include in enumerate(mask):
                    if include:
                        filtered_data.append(data[i])
                data = filtered_data


        self.av_files = []
        for item in data:
            self.av_files.append(item)

        self.preprocess_train = create_transform(
                input_size = 256,
                is_training=True,
                color_jitter = True,
                auto_augment = None,
                interpolation = "bicubic",
                re_prob = 0,
                re_mode = 0,
                re_count = "const",
                mean = (0.485, 0.456, 0.406),
                std = (0.229, 0.224, 0.225),
            )

        self.preprocess_test = transforms.Compose(
                [
                    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.CenterCrop(256),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )

        self.skip_norm = True
        self.noise = False

    def __len__(self): 
        return len(self.av_files)

    def __getitem__(self, idx):
        av_file = self.av_files[idx]

        # Text
        token_path = os.path.join(self.text_feature_path, av_file + '_token.npy')
        text_token = np.load(token_path)
        text_token = torch.tensor(text_token)

        # Visual
        image_path = os.path.join(self.visual_feature_path, av_file + ".jpg.npy")
        image = np.load(image_path)
        image = torch.tensor(image) 

        label = self.classes.index(self.data2class[av_file])
        if 'qmf' in self.args.model_type or 'lreg' in self.args.model_type:
            return text_token, image, label, idx
        return text_token, image, label

def get_data(args):
    train_dataset = MultimodalFoodDataset(args, mode='train')
    val_dataset = MultimodalFoodDataset(args, mode='dev')
    test_dataset = MultimodalFoodDataset(args, mode='test')

    return train_dataset, val_dataset, test_dataset

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    setattr(args, "data_path", "../data/food101/")
    dataset = MultimodalFoodDataset(args, mode='train')

    train_loader = DataLoader(
        dataset, 
        batch_size = 32
    )

    batch = next(iter(train_loader))

    print(batch[1].shape)
    