"""Implements datasets for ENRICO dataset."""


from torch.utils.data import Dataset, WeightedRandomSampler
import random
import csv
import os
import json
from collections import Counter

import torch
from torchvision import transforms

from PIL import Image
import numpy as np

def _add_screen_elements(tree, element_list):
    """Helper function for extracting UI elements from hierarchy. (unused?)"""
    if 'children' in tree and len(tree['children']) > 0:
        # we are at an intermediate node
        for child in tree['children']:
            _add_screen_elements(child, element_list)
    else:
        # we are at a leaf node
        if 'bounds' in tree and 'componentLabel' in tree:
            # valid leaf node
            nodeBounds = tree['bounds']
            nodeLabel = tree['componentLabel']
            node = (nodeBounds, nodeLabel)
            element_list.append(node)


class EnricoDataset(Dataset):
    """Implements torch dataset class for ENRICO dataset."""
    
    def __init__(self, data_dir, mode="train", img_dim_x=128, img_dim_y=256, random_seed=42, train_split=0.65, val_split=0.15, test_split=0.2, normalize_image=False, seq_len=64):
        """Instantiate ENRICO dataset.

        Args:
            data_dir (str): Data directory.
            mode (str, optional): What data to extract. Defaults to "train".
            img_dim_x (int, optional): Image width. Defaults to 128.
            img_dim_y (int, optional): Image height. Defaults to 256.
            random_seed (int, optional): Seed to split dataset on and shuffle data on. Defaults to 42.
            train_split (float, optional): Percentage of training data split. Defaults to 0.65.
            val_split (float, optional): Percentage of validation data split. Defaults to 0.15.
            test_split (float, optional): Percentage of test data split. Defaults to 0.2.
            normalize_image (bool, optional): Whether to normalize image or not Defaults to False.
            seq_len (int, optional): Length of sequence. Defaults to 64.
        """
        super(EnricoDataset, self).__init__()
        self.img_dim_x = img_dim_x
        self.img_dim_y = img_dim_y
        self.seq_len = seq_len
        csv_file = os.path.join(data_dir, "design_topics.csv")
        self.img_dir = os.path.join(data_dir, "screenshots")
        self.wireframe_dir = os.path.join(data_dir, "wireframes")
        self.hierarchy_dir = os.path.join(data_dir, "hierarchies")


        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            example_list = list(reader)

        # the wireframe files are corrupted for these files
        IGNORES = set(["50105", "50109"])
        example_list = [
            e for e in example_list if e['screen_id'] not in IGNORES]

        # length 1458
        self.example_list = example_list

        keys = list(range(len(example_list)))
        # shuffle and create splits
        random.Random(random_seed).shuffle(keys)

        if mode == "train":
            # train split is at the front
            start_index = 0
            stop_index = int(len(example_list) * train_split)
        elif mode == "val":
            # val split is in the middle
            start_index = int(len(example_list) * train_split)
            stop_index = int(len(example_list) * (train_split + val_split))
        elif mode == "test":
            # test split is at the end
            start_index = int(len(example_list) * (train_split + val_split))
            stop_index = len(example_list)

        # only keep examples in the current split
        keys = keys[start_index:stop_index]
        self.keys = keys

        img_transforms = [
            transforms.Resize((img_dim_y, img_dim_x)),
            transforms.ToTensor()
        ]
        if normalize_image:
            img_transforms.append(transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

        # pytorch image transforms
        self.img_transforms = transforms.Compose(img_transforms)

        # make maps
        topics = set()
        for e in example_list:
            topics.add(e['topic'])
        topics = sorted(list(topics))

        idx2Topic = {}
        topic2Idx = {}

        for i in range(len(topics)):
            idx2Topic[i] = topics[i]
            topic2Idx[topics[i]] = i

        self.idx2Topic = idx2Topic
        self.topic2Idx = topic2Idx

        UI_TYPES = ["Text", "Text Button", "Icon", "Card", "Drawer", "Web View", "List Item", "Toolbar", "Bottom Navigation", "Multi-Tab", "List Item", "Toolbar", "Bottom Navigation", "Multi-Tab",
                    "Background Image", "Image", "Video", "Input", "Number Stepper", "Checkbox", "Radio Button", "Pager Indicator", "On/Off Switch", "Modal", "Slider", "Advertisement", "Date Picker", "Map View"]

        idx2Label = {}
        label2Idx = {}

        for i in range(len(UI_TYPES)):
            idx2Label[i] = UI_TYPES[i]
            label2Idx[UI_TYPES[i]] = i

        self.idx2Label = idx2Label
        self.label2Idx = label2Idx
        self.ui_types = UI_TYPES

        # Precompute random x1 mappings
        self.random_x1_mapping = {}
        topics_by_screen_id = {e['screen_id']: e['topic'] for e in self.example_list}
        screen_ids_by_topic = {}
        for e in self.example_list:
            if e['topic'] in screen_ids_by_topic:
                screen_ids_by_topic[e['topic']].append(e['screen_id'])
            else:
                screen_ids_by_topic[e['topic']] = [e['screen_id']]
        
        for screen_id, topic in topics_by_screen_id.items():
            possible_screens = [sid for t, sids in screen_ids_by_topic.items() if t != topic for sid in sids]
            self.random_x1_mapping[screen_id] = random.choice(possible_screens)

    def __len__(self):
        """Get number of samples in dataset."""
        return len(self.keys)

    def featurizeElement(self, element):
        """Convert element into tuple of (bounds, one-hot-label)."""
        bounds, label = element
        labelOneHot = [0 for _ in range(len(self.ui_types))]
        labelOneHot[self.label2Idx[label]] = 1
        return bounds, labelOneHot

    def __getitem__(self, idx):
        """Get item in dataset with efficiently mixed up x1-label pairings and intact x2-label pairings.

        Args:
            idx (int): Index of data to get.

        Returns:
            list: List of (screen image, screen wireframe image, screen label) with efficiently mixed x1-label pairings.
        """

        original_example = self.example_list[self.keys[idx]]

        # Decide whether to use the original or a random x1 image for the current example
        use_random_x1 = random.random() > 0.9 # use 1/20 for complete random
        selected_screen_id = self.random_x1_mapping[original_example['screen_id']] if use_random_x1 else original_example['screen_id']

        # x1 image modality (screen image)
        screenImg = Image.open(os.path.join(
            self.img_dir, selected_screen_id + ".jpg")).convert("RGB")
        screenImg = self.img_transforms(screenImg)

        # x2 image modality (screen wireframe image) and label remain paired with the original example
        screenWireframeImg = Image.open(os.path.join(
            self.wireframe_dir, original_example['screen_id'] + ".png")).convert("RGB")
        screenWireframeImg = self.img_transforms(screenWireframeImg)
        screenLabel = self.topic2Idx[original_example['topic']]

        return [screenImg, screenWireframeImg, screenLabel]



def get_data(data_dir):
    """Get dataloaders for this dataset.

    Args:
        data_dir (str): Data directory.

    Returns:
        tuple: Tuple of ((train dataset, validation dataset, test dataset, sampler))
    """
    ds_train = EnricoDataset(data_dir, mode="train")
    ds_val = EnricoDataset(data_dir, mode="val")
    ds_test = EnricoDataset(data_dir, mode="test")

    targets = []
    class_counter = Counter()
    for i in range(len(ds_train)):
        example_topic = ds_train.example_list[ds_train.keys[i]]['topic']
        targets.append(example_topic)
        class_counter[example_topic] += 1

    weights = []
    for t in targets:
        weights.append(1 / class_counter[t])

    weights = torch.tensor(weights, dtype=torch.float)
    sampler = WeightedRandomSampler(weights, len(targets))

    return (ds_train, ds_val, ds_test, sampler)
