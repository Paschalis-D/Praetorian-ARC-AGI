from torch.utils.data import DataLoader
from models.refinement_model import *
from torch_datasets.refinement_dataset import RefinementDataset
from models.cycleGAN import *
from models.relation_network import *
from tqdm import tqdm
import torch
import os
import json
import numpy as np


class TrainRefinement:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Hyper-parameters
        self.epochs = 20
        self.batch_size = 1
        self.dataloader_workers = 4
        self.learning_rate = 0.0001
        self.adam_betas = (0.5, 0.999)
        self.refinement_threshold = 0.01  # Threshold for stopping refinement
        self.max_refinement_steps = 20  # Maximum refinement steps
        self.noise_level = 0.1  # Noise level for variability
        
        # Directories
        self.data_dir = "D:/Praetorian-ARC-AGI/arc-prize"
        self.model_save_dir = "D:/Praetorian-ARC-AGI/trained_models"
        self.output_save_dir = "D:/Praetorian-ARC-AGI/"
        
        # Models
        self.model = RefinementNetwork(num_unets=3).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, betas=self.adam_betas)
        
        self.gan = GeneratorResNet(input_channels=1, n_residual_blocks=15)
        self.gan.load_state_dict(torch.load(os.path.join(self.model_save_dir, "generator_xy.pth"), map_location=self.device, weights_only=True))
        self.gan = self.gan.to(self.device)

        self.relational = RelationNetwork()
        self.relational.load_state_dict(torch.load(os.path.join(self.model_save_dir, "relational.pth"), map_location=self.device, weights_only=True))
        self.relational = self.relational.to(self.device)

        # Data loader
        self.dataloader = DataLoader(
            RefinementDataset(self.data_dir, self.device),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.dataloader_workers,
        )
    
    def run(self):
        print("Running refinement process.")
        self.model.train()
        all_outputs = {}

        for epoch in range(self.epochs):
            running_loss = 0.0
            train_loader = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.epochs}", leave=False)

            for i, batch in enumerate(train_loader):
                input_grid = batch["x"].to(self.device)
                task_outputs = [t.to(self.device) for t in batch["task_outputs"]]
                task_id = batch["task_id"]

                if isinstance(task_id, list):
                    task_id = task_id[0]  # If task_id is a list, get the first element

                # Start with the initial refinement input from the GAN
                refinement_input = self.gan(input_grid) + torch.randn_like(input_grid) * self.noise_level
                
                self.optimizer.zero_grad()
                all_outputs[task_id] = []

                for step in range(self.max_refinement_steps):
                    # Forward pass through the refinement network
                    refinement_output = self.model(refinement_input)

                    # Calculate the relational loss
                    relation_scores = [self.relational(refinement_output, task_output) for task_output in task_outputs]
                    avg_relation_score = torch.mean(torch.stack(relation_scores))

                    # If the relational score is below the threshold, stop refining
                    if avg_relation_score < self.refinement_threshold:
                        break

                    # Reverse transformation before saving the attempt
                    attempt_output = self.reverse_transform(refinement_output.squeeze(0).detach().cpu().numpy())
                    all_outputs[task_id].append({f"attempt_{step+1}": attempt_output})

                    # Use the relational loss as the training loss
                    training_loss = avg_relation_score
                    training_loss.backward()
                    self.optimizer.step()

                    # Update refinement input for the next iteration
                    refinement_input = refinement_output.detach()

                running_loss += avg_relation_score.item()
                train_loader.set_postfix({"loss": running_loss / (i + 1)})

            print(f"Epoch {epoch+1} complete with average loss: {running_loss / len(self.dataloader)}")

        # Save final outputs after all epochs
        self.save_final_outputs(all_outputs)




    def reverse_transform(self, output):
        # Remove padding (pixels with a value of -1)
        output = output[output != -1].reshape(-1, output.shape[-1])

        # Multiply by 9 and round to the nearest integer
        output = np.round(output * 9).astype(int)

        return output.tolist()

    def save_final_outputs(self, outputs):
        os.makedirs(self.output_save_dir, exist_ok=True)
        output_file = os.path.join(self.output_save_dir, "final_outputs.json")
        with open(output_file, "w") as f:
            json.dump(outputs, f)
        print(f"Final outputs saved to {output_file}")



