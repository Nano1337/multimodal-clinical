import pandas as pd
import warnings
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from transformers import BertTokenizer
from sklearn.preprocessing import LabelEncoder
import re
import argparse

# Constants
max_length = 512  # Adjust based on your specific needs
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # Adjust model as needed

class MultimodalFoodDataset(Dataset):
    def __init__(self, args, mode="train"):
        """
        Args:
            args: Arguments containing data_path.
            mode (string): "train" or "test" to specify the dataset to load.
        """
        self.mode = mode
        self.image_dir = os.path.join(args.data_path, "images/")
        self.csv_file = os.path.join(args.data_path, f"texts/{mode}_titles.csv")
        self.data_frame = pd.read_csv(self.csv_file, names=['image_path', 'text', 'food'], header=None)

        # label -> long encoding
        self.label_encoder = LabelEncoder()
        self.data_frame['encoded_food'] = self.label_encoder.fit_transform(self.data_frame['food'])
        self.labels = self.data_frame['encoded_food'].tolist()

        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            
            transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
            # transforms.RandomRotation(10),  # Randomly rotate the image by +/- 10 degrees
            # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Randomly translate the image
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Randomly alter the brightness, contrast, saturation, and hue
            # transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.33)),  # Randomly crop the image then resize it back to the given size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_name = self.data_frame.iloc[idx, 0]
        image_path = os.path.join(self.image_dir, self.mode, extract_class_name(image_name), image_name)
        image = Image.open(image_path).convert('RGB')  # Ensure image is RGB
        if self.mode == "train": 
            image = self.train_transform(image)
        else: 
            image = self.test_transform(image)
            
        text = self.data_frame.iloc[idx, 1]
        text = preprocess_text(text)

        # Corrected line
        label = self.data_frame['encoded_food'].iloc[idx]  # Use encoded label

        return image, text, label

    def balanced_sampler(self):
        class_counts = torch.bincount(torch.tensor(self.labels))
        class_weights = 1. / class_counts
        sample_weights = class_weights[self.labels]
        sampler = WeightedRandomSampler(sample_weights, len(self.labels), replacement=True)
        return sampler  

def extract_class_name(filename): 
    parts = filename.split("_")
    class_label_parts = parts[:-1]
    return '_'.join(class_label_parts)



def preprocess_text(text):
    """Preprocesses the text as per BERT requirements"""
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub('[^a-zA-Z]', ' ', text)  # Remove punctuations and numbers
    text = re.sub(r"\s+[a-zA-Z]\s+", ' ', text)  # Single character removal
    text = re.sub(r'\s+', ' ', text)  # Removing multiple spaces
    text = text.lower()
    return text

def custom_collate_fn(batch):
    images, texts, labels = zip(*batch)
    images = torch.stack(images, dim=0)
    encoded_texts = tokenizer(list(texts), padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    labels = torch.tensor(labels, dtype=torch.long)  # Ensure labels are torch.long

    return images, encoded_texts, labels

def get_data(args): 
    train_set = MultimodalFoodDataset(args, mode="train")
    test_set = MultimodalFoodDataset(args, mode="test")
    val_set = test_set

    return train_set, val_set, test_set

# Example usage
if __name__ == "__main__":
    dirpath = "../data/food101"
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    setattr(args, 'data_path', dirpath)

    # Load dataset
    dataset = MultimodalFoodDataset(args, mode="train")
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=custom_collate_fn, sampler=dataset.balanced_sampler())

    batch = next(iter(dataloader))

    print("x1", batch[0].shape)
    print("x2", batch[1]['input_ids'].shape)
    print("labels", batch[2].shape)
    print(f'Labels: {batch[2]}')