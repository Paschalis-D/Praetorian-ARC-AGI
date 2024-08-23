from torch.utils.data import DataLoader
from models.relation_network import *
from torch_datasets.relation_dataset import RelationDataset
from tqdm import tqdm
import torch
import os

class TrainRelational:
    def __init__(self):
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Hyper-parameters
        self.epochs = 20
        self.batch_size = 1
        self.dataloader_workers = 4
        self.learning_rate = 0.0001
        self.adam_betas = (0.5, 0.999)
        self.decay_start = 5

        # The paper suggests using a least-squares loss instead of
        # negative log-likelihood, at it is found to be more stable.
        self.loss = torch.nn.MSELoss()

        # Image dimensions
        self.img_height = 32
        self.img_width = 32
        self.img_channels = 1

        # Models
        self.model: RelationNetwork

        # Optimizers
        self.optimizer: torch.optim.Adam

        # Data loaders
        self.data_dir = "D:/Praetorian-ARC-AGI/arc-prize"
        self.dataloader: DataLoader
        self.valid_dataloader: DataLoader

        # Track the best validation loss
        self.best_val_loss = float('inf')

    def initialize(self):
        """
        ## Initialize models and data loaders
        """
        print("Initializing trining...")
        # Create the models
        self.model = RelationNetwork().to(self.device)
        self.model.load_state_dict(torch.load("./checkpoints/relational_checkpoint.pth", map_location=torch.device(self.device), weights_only=True))
        print("Model loaded.")
        # Create the optmizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, betas=self.adam_betas)
        print(f"Optimizer was set with {self.learning_rate} learning rate and {self.adam_betas} betas.")

        # Training data loader
        self.dataloader = DataLoader(
            RelationDataset(self.data_dir, "train"),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.dataloader_workers,
        )

        # Validation data loader
        self.valid_dataloader = DataLoader(
            RelationDataset(self.data_dir, "val"),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.dataloader_workers,
        )
        print(f"Dataloaders were set with {self.batch_size} batch size and {self.dataloader_workers} workers.")

    def run(self):
        print("Training Relational Netwrok on arc-agi data.")
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0

            # Using tqdm to show the progress
            train_loader = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.epochs}", leave=False)

            for i, batch in enumerate(train_loader):
                output1, output2, label = batch
                output1, output2, label = output1.to(self.device), output2.to(self.device), label.to(self.device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                relation = self.model(output1, output2)

                # Compute the loss
                loss = self.loss(relation, label)
                running_loss += loss.item()

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

                # Update tqdm with current loss
                train_loader.set_postfix({"loss": running_loss / (i + 1)})

        print("Training complete.")

    def evaluate(self):
        """
        Evaluate the model on the validation set
        """
        print("Evaluating the Relational Network on the validation set")
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():    
            for batch in self.valid_dataloader:
                output1, output2, label = batch
                output1, output2, label = output1.to(self.device), output2.to(self.device), label.to(self.device)

                relation = self.model(output1, output2)
                loss = self.loss(relation, label)
                running_loss += loss.item()  # Use .item() to accumulate the scalar loss
            
        print(f"Validation loss: {running_loss / len(self.valid_dataloader)}")

    def save_model(self, epoch, val_loss):
        """
        Save the model state_dict with the current epoch and validation loss
        """
        model_save_path = os.path.join(f"model_epoch_{epoch+1}_val_loss_{val_loss:.4f}.pth")
        torch.save(self.model.state_dict(), model_save_path)
        print(f"Model saved as {model_save_path}")
