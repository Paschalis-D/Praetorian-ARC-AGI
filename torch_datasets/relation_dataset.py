import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import random
from itertools import combinations

import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from itertools import combinations
import random

class RelationDataset(Dataset):
    def __init__(self, data_dir: str, split: str):
        if split not in ("train", "val", "test"):
            raise ValueError(f"{split} is not a valid split for the RelationDataset.")
        
        self.data_dir = data_dir
        self.split = split
        
        # Load the appropriate JSON file based on the split
        if split == "train":
            json_filename = "arc-agi_training_challenges.json"
        elif split == "val":
            json_filename = "arc-agi_evaluation_challenges.json"
        elif split == "test":
            json_filename = "arc-agi_test_challenges.json"

        # Initialize a dictionary to store tasks and their outputs
        self.tasks = {}

        # Load and process the selected JSON file
        json_path = os.path.join(data_dir, json_filename)
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        # Always extract the data from the "train" key
        for task_id, task_data in data.items():
            outputs = []
            for item in task_data.get("train", []):  # Always get data from "train"
                input_grid = torch.tensor(item["input"], dtype=torch.float32)
                output_grid = torch.tensor(item["output"], dtype=torch.float32)

                # Apply preprocessing: padding, normalization, etc.
                input_grid = self.preprocess(input_grid)
                output_grid = self.preprocess(output_grid)

                if input_grid.shape[1] <= 32 and input_grid.shape[2] <= 32 and \
                   output_grid.shape[1] <= 32 and output_grid.shape[2] <= 32:
                    outputs.append(output_grid)
                else:
                    print(f"Skipping example with size {input_grid.shape} and {output_grid.shape}.")
            if outputs:
                self.tasks[task_id] = outputs

        # Prepare list of task ids
        self.task_ids = list(self.tasks.keys())

        # Generate pairs
        self.same_task_pairs = []
        self.different_task_pairs = []

        for task_id, outputs in self.tasks.items():
            # Generate all combinations of pairs within the same task
            pairs = list(combinations(outputs, 2))
            self.same_task_pairs.extend([(p1, p2, torch.tensor([0], dtype=torch.float32)) for p1, p2 in pairs])

        # Now, generate an equal number of different-task pairs
        total_same_task_pairs = len(self.same_task_pairs)
        while len(self.different_task_pairs) < total_same_task_pairs:
            task_id1, task_id2 = random.sample(self.task_ids, 2)
            output1 = random.choice(self.tasks[task_id1])
            output2 = random.choice(self.tasks[task_id2])
            self.different_task_pairs.append((output1, output2, torch.tensor([1], dtype=torch.float32)))

        # Combine and shuffle pairs
        self.all_pairs = self.same_task_pairs + self.different_task_pairs
        random.shuffle(self.all_pairs)

    def preprocess(self, grid):
        """
        Preprocess the grid:
        - Normalize the pixel values.
        - Pad the grid to a 32x32 grid.
        """
        grid = grid.unsqueeze(0)  # Add channel dimension
        grid = grid / 9.0  # Normalize pixel values
        grid = self.pad(grid, 32)  # Pad to 32x32 grid
        return grid

    def pad(self, tensor, size):
        """
        Pad the tensor to the desired size.
        """
        current_height = tensor.size(1)
        current_width = tensor.size(2)

        pad_height = size - current_height
        pad_width = size - current_width

        padding = (pad_width // 2, pad_width - pad_width // 2, pad_height // 2, pad_height - pad_height // 2)
        padded_tensor = F.pad(tensor, padding, mode='constant', value=-1)
        return padded_tensor

    def __len__(self):
        return len(self.all_pairs)

    def __getitem__(self, idx):
        return self.all_pairs[idx]

# Example usage
if __name__ == "__main__":
    data_dir = "./arc-prize"
    dataset = RelationDataset(data_dir, split="train")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    for batch in dataloader:
        output1, output2, labels = batch
        print("Output1 shape:", output1.shape)
        print("Output2 shape:", output2.shape)
        print("Labels:", labels)
        break

