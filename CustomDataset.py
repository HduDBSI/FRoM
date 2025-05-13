from torch.utils.data import Dataset, Subset
import torch
import json
from transformers import AutoTokenizer
import numpy as np
from copy import deepcopy

class CustomDataset(Dataset):
    def __init__(self, texts, labels, max_length=128):
        tokenizer = AutoTokenizer.from_pretrained('roberta/')
        outputs = tokenizer(texts, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')
        self.input_ids = outputs['input_ids']
        self.attention_mask = outputs['attention_mask']
        self.label = torch.tensor(labels)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):

        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'label': self.label[idx],
            'idx': idx 
        }

def read_data(jsonl_file):
    texts = []
    labels = []

    with open(jsonl_file, "r") as file:
        for line in file:
            sample = json.loads(line)
            texts.append(sample["comment_text"])
            labels.append(sample["label"])

    return texts, labels

def build_dataset(jsonl_file, max_length=512, class_num=4):
    texts, labels = read_data(jsonl_file)
    if class_num == 2:
        labels = [int(label != 0) for label in labels]
    dataset = CustomDataset(texts, labels, max_length)
    return dataset

# Random Under-Sampling
def build_balanced_dataset(dataset):
    # Get sample indices for each class
    labels = dataset.label.numpy()
    class_indices = {label: np.where(labels == label)[0] for label in np.unique(labels)}
    
    # Find the minimum number of samples across all classes
    min_class_size = min(len(indices) for indices in class_indices.values())

    # Perform under-sampling for each class to ensure each class has the same number of samples as the minor class
    balanced_indices = []
    for indices in class_indices.values():
        balanced_indices.extend(np.random.choice(indices, min_class_size, replace=False))

    # Create a balanced dataset using the selected indices
    balanced_dataset = Subset(dataset, balanced_indices)
    return balanced_dataset
