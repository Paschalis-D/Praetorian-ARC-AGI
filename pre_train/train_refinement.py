from torch.utils.data import DataLoader
from models.refinement_model import *
from torch_datasets.refinement_dataset import RefinementDataset
from models.cycleGAN import *
from models.relation_network import *
from tqdm import tqdm
import torch
import os

class TrainRefinement:
    def __init__(self):
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Hyper-parameters
        self.epochs = 20
        self.batch_size = 1
        self.dataloader_workers = 4
        self.learning_rate = 0.0001
        self.adam_betas = (0.5, 0.999)
        self.decay_start = 5
        self.refinement_threshold = 0.01  # Set your threshold here
        self.max_refinement_steps = 10  # Limit the number of refinement steps

        # The paper suggests using a least-squares loss instead of
        # negative log-likelihood, as it is found to be more stable.
        self.loss = torch.nn.MSELoss()

        # Image dimensions
        self.img_height = 32
        self.img_width = 32
        self.img_channels = 1

        # Models
        self.model: RefinementNetwork

        # Optimizers
        self.optimizer: torch.optim.Adam

        # Data loaders
        self.data_dir = "D:/Praetorian-ARC-AGI/arc-prize"
        self.dataloader: DataLoader
        self.valid_dataloader: DataLoader

        # Track the best validation loss
        self.best_val_loss = float('inf')

    def initialize(self):
        print("Initializing training...")

        # Create the models
        self.model = RefinementNetwork(num_unets=3).to(self.device)
        print("Model loaded.")
        # Create the optimizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, betas=self.adam_betas)
        print(f"Optimizer was set with {self.learning_rate} learning rate and {self.adam_betas} betas.")

        self.gan = GeneratorResNet(input_channels=1, n_residual_blocks=15)
        self.gan.load_state_dict(torch.load("./trained_models/generator_xy.pth", map_location=torch.device(self.device)))

        self.relational = RelationNetwork()
        self.relational.load_state_dict(torch.load("./trained_models/relational.pth", map_location=torch.device(self.device)))

        # Training data loader
        self.dataloader = DataLoader(
            RefinementDataset(self.data_dir, self.device, "train"),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.dataloader_workers,
        )

        # Validation data loader
        self.valid_dataloader = DataLoader(
            RefinementDataset(self.data_dir, self.device, "val"),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.dataloader_workers,
        )
        print(f"Dataloaders were set with {self.batch_size} batch size and {self.dataloader_workers} workers.")

    def run(self):
        print("Training Relational Network on arc-agi data.")
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0

            # Using tqdm to show the progress
            train_loader = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.epochs}", leave=False)

            for i, batch in enumerate(train_loader):
                input = batch["x"].to(self.device)
                output = batch["y"].to(self.device)
                task_outputs = batch["task_outputs"]  # Task-specific outputs from dataset
                task_outputs = [t.to(self.device) for t in task_outputs]

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                refinement_input = self.gan(input)
                relation_input = refinement_input
                for step in range(self.max_refinement_steps):
                    # Forward pass through the refinement model
                    relation_input = self.model(relation_input)
                    
                    # Calculate the relational network output for all task outputs
                    relation_scores = [self.relational(relation_input, task_output) for task_output in task_outputs]
                    avg_relation_score = torch.mean(torch.stack(relation_scores))

                    if avg_relation_score < self.refinement_threshold:
                        break

                # Calculate final loss between the refined output and ground truth `y`
                final_loss = self.loss(relation_input, output)
                running_loss += final_loss.item()

                # Backward pass and optimize
                final_loss.backward()
                self.optimizer.step()

                # Update tqdm with current loss
                train_loader.set_postfix({"loss": running_loss / (i + 1)})

            print(f"Epoch {epoch+1} complete.")

    def evaluate(self):
        """
        Evaluate the model on the validation set
        """
        print("Evaluating the Relational Network on the validation set")
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():    
            for batch in self.valid_dataloader:
                input = batch["x"].to(self.device)
                output = batch["y"].to(self.device)
                task_outputs = batch["task_outputs"]  # Task-specific outputs from dataset
                task_outputs = [t.to(self.device) for t in task_outputs]

                refinement_input = self.gan(input)
                relation_input = refinement_input
                for step in range(self.max_refinement_steps):
                    relation_input = self.model(relation_input)
                    relation_scores = [self.relational(relation_input, task_output) for task_output in task_outputs]
                    avg_relation_score = torch.mean(torch.stack(relation_scores))

                    if avg_relation_score < self.refinement_threshold:
                        break

                loss = self.loss(relation_input, output)
                running_loss += loss.item()

        avg_loss = running_loss / len(self.valid_dataloader)
        print(f"Validation loss: {avg_loss}")

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "refinement.pth")
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")