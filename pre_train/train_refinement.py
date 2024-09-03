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
        self.batch_size = 1
        self.dataloader_workers = 4
        self.learning_rate = 0.00001
        self.adam_betas = (0.5, 0.999)
        self.refinement_threshold = 0.1  # Threshold for stopping refinement
        self.max_refinement_steps = 20  # Maximum refinement steps
        self.supervised_loss_weight = 0.7  # Weight for supervised loss
        self.relational_loss_weight = 0.3  # Weight for relational loss
        
        # Directories
        self.data_dir = "D:/Praetorian-ARC-AGI/arc-prize"
        self.model_save_dir = "D:/Praetorian-ARC-AGI/trained_models"
        self.output_save_dir = "D:/Praetorian-ARC-AGI/"
        
        # Models
        self.model = RefinementNetwork(num_unets=3).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, betas=self.adam_betas)
        self.supervised_loss_fn = torch.nn.MSELoss()  # Supervised loss function
        
        self.gan = GeneratorResNet(input_channels=1, n_residual_blocks=15)
        self.gan.load_state_dict(torch.load(os.path.join(self.model_save_dir, "generator_xy.pth"), map_location=self.device, weights_only=True))
        self.gan = self.gan.to(self.device)

        self.relational = RelationNetwork()
        self.relational.load_state_dict(torch.load(os.path.join(self.model_save_dir, "relational.pth"), map_location=self.device, weights_only=True))
        self.relational = self.relational.to(self.device)

        # Data loaders
        self.train_dataloader = DataLoader(
            RefinementDataset(self.data_dir, self.device, split="train"),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.dataloader_workers,
        )
        self.test_dataloader = DataLoader(
            RefinementDataset(self.data_dir, self.device, split="test"),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.dataloader_workers,
        )

    def supervised_training(self):
        print("Starting supervised training on labeled data.")
        self.model.train()

        for epoch in range(10):  # Number of epochs for supervised training can be adjusted
            running_loss = 0.0
            train_loader = tqdm(self.train_dataloader, desc=f"Supervised Epoch {epoch+1}", leave=False)

            for i, batch in enumerate(train_loader):
                input_grid = batch["x"].to(self.device)
                target_grid = batch["y"].to(self.device)
                task_outputs = [t.to(self.device) for t in batch["task_outputs"]]

                self.optimizer.zero_grad()

                # Forward pass through the refinement network
                output = self.model(input_grid)

                # Compute the supervised loss
                supervised_loss = self.supervised_loss_fn(output, target_grid)

                # Compute the relational loss
                relational_scores = [self.relational(output, task_output) for task_output in task_outputs]
                relational_loss = torch.mean(torch.stack(relational_scores))

                # Combine the losses
                total_loss = (self.supervised_loss_weight * supervised_loss) + (self.relational_loss_weight * relational_loss)
                total_loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                running_loss += total_loss.item()
                train_loader.set_postfix({"loss": running_loss / (i + 1)})

            print(f"Supervised Epoch {epoch+1} complete with average loss: {running_loss / len(self.train_dataloader)}")

    def refinement_phase(self):
        print("Starting refinement phase on test data.")
        self.model.train()  # Set the model to training mode for refinement
        all_outputs = {}

        for i, batch in enumerate(tqdm(self.test_dataloader, desc="Refinement Phase")):
            input_grid = batch["x"].to(self.device)
            task_outputs = [t.to(self.device) for t in batch["task_outputs"]]
            task_id = batch["task_id"]

            if isinstance(task_id, list):
                task_id = task_id[0]  # If task_id is a list, get the first element

            print(f"Processing task: {task_id}")

            # Start with the initial refinement input from the GAN
            refinement_input = self.gan(input_grid)
            
            final_output = None

            for step in range(self.max_refinement_steps):
                # Forward pass through the refinement network
                refinement_output = self.model(refinement_input)

                # Calculate the relational loss
                relation_scores = [self.relational(refinement_output, task_output) for task_output in task_outputs]
                avg_relation_score = torch.mean(torch.stack(relation_scores))

                # Log the current refinement step and relation score
                print(f"  Refinement step {step + 1}/{self.max_refinement_steps}, Relation score: {avg_relation_score.item():.4f}")

                # If the relational score is below the threshold or is `nan`, stop refining
                if avg_relation_score < self.refinement_threshold and avg_relation_score > -self.refinement_threshold:
                    print(f"  Stopping refinement early at step {step + 1} due to low or invalid relation score.")
                    break

                # Backpropagate the relational loss and update the model
                self.optimizer.zero_grad()
                avg_relation_score.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Gradient clipping to prevent exploding gradients
                self.optimizer.step()

                # Store the final output after the last step
                final_output = refinement_output

                # Update refinement input for the next iteration
                refinement_input = refinement_output.detach()

                # Reverse transformation and save the final output for this task
            if final_output is not None:
                attempt_output = self.reverse_transform(final_output.squeeze(0).detach().cpu().numpy())
                all_outputs[task_id] = [{"attempt_1": attempt_output, "attempt_2": attempt_output}]

        # Save final outputs
        self.save_final_outputs(all_outputs)

    def reverse_transform(self, output):
        # Remove padding (pixels with a value of -1 or any negative value)
        output = output[output >= 0]

        # Check if output is empty
        if len(output) == 0:
            print("Warning: All values in the output grid are negative or zero.")
            return []  # Return an empty list to indicate no valid output

        # Multiply by 9 and round to the nearest integer
        output = np.round(output * 9).astype(int)

        # Determine the original grid shape before padding
        height, width = self.determine_original_shape(len(output))

        # Reshape to the determined original grid shape
        output = output[:height * width].reshape((height, width))

        return output.tolist()

    def determine_original_shape(self, num_pixels):
        if num_pixels == 0:
            return 0, 0  # Handle the edge case where there are no valid pixels

        # Try to find a pair of factors of num_pixels that are close to each other
        factors = [(i, num_pixels // i) for i in range(1, int(np.sqrt(num_pixels)) + 1) if num_pixels % i == 0]
        if not factors:
            raise ValueError("No valid factors found for determining grid shape.")

        # Select the pair of factors that has the smallest difference, which is most "square-like"
        height, width = min(factors, key=lambda x: abs(x[0] - x[1]))
        return height, width


    def save_final_outputs(self, outputs):
        os.makedirs(self.output_save_dir, exist_ok=True)
        output_file = os.path.join(self.output_save_dir, "final_outputs.json")
        with open(output_file, "w") as f:
            json.dump(outputs, f)
        print(f"Final outputs saved to {output_file}")

    def run(self):
        # Perform supervised training on the training dataset
        self.supervised_training()

        # Then, perform the refinement phase on the test dataset
        self.refinement_phase()






