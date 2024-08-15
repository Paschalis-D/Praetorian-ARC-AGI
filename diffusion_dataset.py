import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch.nn.functional as F


class DiffusionDataset(Dataset):
    def __init__(self, json_dir, device, task, split=None):
        self.json_dir = json_dir
        self.device = device
        self.split = split  # Can be 'train', 'val', or 'test'
        self.train_examples = []
        self.dataset = pd.read_json(self.json_dir)
        self.task = self.dataset.columns[task]

        train_ex = self.dataset[self.task][1]
        for example in train_ex:
            self.train_examples.append(example)
                
    def __len__(self):
        return len(self.train_examples)

    def __getitem__(self, idx):
        try:
            example = self.train_examples[idx]
            input = torch.tensor(example["input"], dtype=torch.float32).unsqueeze(0)
            label = torch.tensor(example["output"], dtype=torch.float32).unsqueeze(0)

            # Normalize the pixel values
            input = input / 9.0
            label = label / 9.0

            # Pad the images to a 30x30 grid
            input = self.pad_to_30x30(input)
            label = self.pad_to_30x30(label)

            # Move tensors to the specified device
            input = input.to(self.device)
            label = label.to(self.device)

            return input, label

        except Exception as e:
            print(f"An error occurred while processing index {idx}: {e}")
            raise

    def pad_to_30x30(self, tensor):
        current_height = tensor.size(1)
        current_width = tensor.size(2)

        pad_height = 30 - current_height
        pad_width = 30 - current_width

        padding = (pad_width // 2, pad_width - pad_width // 2, pad_height // 2, pad_height - pad_height // 2)

        padded_tensor = F.pad(tensor, padding, mode='constant', value=-1)

        return padded_tensor


# Example usage: Load and print the first batch of the training dataset
if __name__ == "__main__":
    json_dir = "D:/ARC/arc-data/arc-agi_training_challenges.json"
    split = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        task = 1
        train_dataset = DiffusionDataset(json_dir, device, task, split)
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        print(len(train_dataloader))
        inputs_list=[]
        labels_list=[]
        for batch in train_dataloader:
            inputs, labels = batch
            inputs_list.append(inputs)
            labels_list.append(labels)
            #print("Input: ", inputs)
            #print("Label: ", labels)
            print(inputs.shape, labels.shape)

        print(len(inputs_list))
        print(len(labels_list))

    except Exception as e:
        print(f"An error occurred: {e}")
