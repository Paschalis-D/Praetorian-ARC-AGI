import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class RefinementDataset(Dataset):
    def __init__(self, data_dir: str, device: torch.device):
        self.data_dir = data_dir
        self.device = device
        
        # Initialize empty lists to store examples and task IDs
        self.examples = []
        self.task_ids = []
        self.task_examples = []

        # Load the JSON data for challenges
        json_filename = "arc-agi_test_challenges.json"
        with open(os.path.join(data_dir, json_filename), "r") as f:
            data = json.load(f)

        # Extract data from the "test" key for each task
        for task_id, task_data in data.items():
            examples = task_data.get("test", [])  # Use the "test" key
            
            for item in examples:
                input_grid = torch.tensor(item["input"], dtype=torch.float32)

                if input_grid.shape[0] <= 32 and input_grid.shape[1] <= 32:
                    self.examples.append(item)
                    self.task_ids.append(task_id)  # Store task ID

                    # Add all outputs for this task from the "train" key as task examples
                    task_train_examples = task_data.get("train", [])
                    task_outputs = [torch.tensor(example["output"], dtype=torch.float32) for example in task_train_examples]
                    self.task_examples.append(task_outputs)
                else:
                    print(f"Skipping example with size {input_grid.shape}.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        try:
            input_grid = torch.tensor(self.examples[idx]["input"], dtype=torch.float32).unsqueeze(0)

            # Normalize the pixel values
            input_grid = input_grid / 9.0

            # Pad the grids to a 32x32 grid
            input_grid = self.pad(input_grid, 32)

            # Task-specific outputs for relational network
            task_outputs = [self.pad(output.unsqueeze(0) / 9.0, 32) for output in self.task_examples[idx]]

            # Move tensors to the specified device
            input_grid = input_grid.to(self.device)
            task_outputs = [output.to(self.device) for output in task_outputs]

            return {"x": input_grid, "task_outputs": task_outputs, "task_id": self.task_ids[idx]}  # Single task ID

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
        train_dataset = RefinementDataset(data_dir, device, split="val")
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