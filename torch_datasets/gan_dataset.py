import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class GanDataset(Dataset):
    def __init__(self, data_dir: str, device: torch.device, split: str):
        self.data_dir = data_dir
        self.device = device
        
        # Initialize empty list to store examples
        self.examples = []

        # Load the appropriate JSON file based on the split
        if split == "train":
            json_filename = "arc-agi_training_challenges.json"
        elif split == "val":
            json_filename = "arc-agi_evaluation_challenges.json"
        elif split == "test":
            json_filename = "arc-agi_test_challenges.json"
        else:
            raise ValueError(f"Invalid split: {split}. Expected 'train', 'val', or 'test'.")

        # Load the JSON data
        with open(os.path.join(data_dir, json_filename), "r") as f:
            data = json.load(f)

        # Extract data from the "train" key for each task
        for task_id, task_data in data.items():
            examples = task_data.get("train", [])  # Always use the "train" key
            
            for item in examples:
                input_grid = torch.tensor(item["input"], dtype=torch.float32)
                output_grid = torch.tensor(item["output"], dtype=torch.float32)

                if input_grid.shape[0] <= 32 and input_grid.shape[1] <= 32 and \
                   output_grid.shape[0] <= 32 and output_grid.shape[1] <= 32:
                    self.examples.append(item)
                else:
                    print(f"Skipping example with size {input_grid.shape} and {output_grid.shape}.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        try:
            example = self.examples[idx]
            
            input_grid = torch.tensor(example["input"], dtype=torch.float32).unsqueeze(0)
            output_grid = torch.tensor(example["output"], dtype=torch.float32).unsqueeze(0)

            # Normalize the pixel values
            input_grid = input_grid / 9.0
            output_grid = output_grid / 9.0

            # Pad the grids to a 30x30 grid
            input_grid = self.pad(input_grid, 32)
            output_grid = self.pad(output_grid, 32)

            # Move tensors to the specified device
            input_grid = input_grid
            output_grid = output_grid

            return {"x": input_grid, 
                    "y": output_grid}

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
    data_dir = "D:/Praetorian-ARC-AGI/arc-prize"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        train_dataset = GanDataset(data_dir, device, split="val")
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        print(len(train_dataset))
        inputs_list=[]
        labels_list=[]
        for batch in train_dataloader:
            inputs, labels = batch["x"], batch["y"]
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
