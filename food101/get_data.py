import pandas as pd
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import BertTokenizer
from sklearn.preprocessing import LabelEncoder
import re
import argparse

# Constants
max_length = 512  # Adjust based on your specific needs
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # Adjust model as needed

class TextImageDataset(Dataset):
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

        # Label encoding
        self.label_encoder = LabelEncoder()
        self.data_frame['encoded_food'] = self.label_encoder.fit_transform(self.data_frame['food'])

        self.transform = transforms.Compose([
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
        image = self.transform(image)
            
        text = self.data_frame.iloc[idx, 1]
        text = preprocess_text(text)

        label = self.data_frame.iloc[idx, 'encoded_food']  # Use encoded label

        return image, text, label

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
    return images, encoded_texts, {'labels': labels}


# Example usage
if __name__ == "__main__":
    dirpath = "../data/food101"
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    setattr(args, 'data_path', dirpath)

    # Load dataset
    dataset = TextImageDataset(args, mode="train")
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=custom_collate_fn)

    batch = next(iter(dataloader))

    print(f'x1: {batch[0].shape}, x2: {batch[1].shape}, label: {batch[2].shape}')

    # for images, texts, labels in dataloader:
    #     print(images.shape, texts['input_ids'].shape, labels.shape)
    #     # This prints the shapes of images, tokenized text input IDs, and labels for each batch
