import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class DiffusionDataset(Dataset):
    def __init__(self, data_dir, device):
        self.data_dir = data_dir
        self.device = device
        self.train_examples = []
        
        # Collect all JSON files in the directory
        self.json_paths = [os.path.join(data_dir, json_file) for json_file in os.listdir(data_dir)]
        
        # Load and process each JSON file
        for json_path in self.json_paths:
            with open(json_path, 'r') as f:
                dataset = json.load(f)
            
            # Extract the 'train' key data
            train_data = dataset.get('train', [])
            for item in train_data:
                self.train_examples.append(item)
                
    def __len__(self):
        return len(self.train_examples)

    def __getitem__(self, idx):
        try:
            example = self.train_examples[idx]
            input_grid = torch.tensor(example["input"], dtype=torch.float32).unsqueeze(0)
            output_grid = torch.tensor(example["output"], dtype=torch.float32).unsqueeze(0)

            # Normalize the pixel values
            input_grid = input_grid / 9.0
            output_grid = output_grid / 9.0

            # Pad the grids to a 30x30 grid
            input_grid = self.pad(input_grid, 30)
            output_grid = self.pad(output_grid, 30)

            # Move tensors to the specified device
            input_grid = input_grid.to(self.device)
            output_grid = output_grid.to(self.device)

            return input_grid, output_grid

        except Exception as e:
            print(f"An error occurred while processing index {idx}: {e}")
            raise

    def pad(self, tensor, size):
        current_height = tensor.size(1)
        current_width = tensor.size(2)

        pad_height = size - current_height
        pad_width = size - current_width

        padding = (pad_width // 2, pad_width - pad_width // 2, pad_height // 2, pad_height - pad_height // 2)

        padded_tensor = F.pad(tensor, padding, mode='constant', value=-1)

        return padded_tensor

# Example usage: Load and print the first batch of the training dataset
if __name__ == "__main__":
    data_dir = "D:/Praetorian-ARC-AGI/arc-all"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        train_dataset = DiffusionDataset(data_dir, device)
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        print(len(train_dataset))
        inputs_list=[]
        labels_list=[]
        for batch in train_dataloader:
            inputs, labels = batch
            inputs_list.append(inputs)
            labels_list.append(labels)
            print("Input: ", inputs)
            print("Label: ", labels)
            print(inputs.shape, labels.shape)
            break

        print(len(inputs_list))
        print(len(labels_list))

    except Exception as e:
        print(f"An error occurred: {e}")
