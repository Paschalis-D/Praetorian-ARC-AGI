# TODO: Create a diffuser model that will be trained in all of ARC data ever existed
# TODO: Update the dataset so that it returns only the specific examples for each task
# TODO: Train the diffuser and save the model
# TODO: Create a transfer-learning loop that will train the diffuser on each task
# TODO: Create the Relational network based on the Learning to Compare: Relation Network for Few-Shot Learning paper.
# TODO: Create a training function for the Relational network and it also might need a new dataset too.
# TODO: Create a training function for the Refinement model.
import argparse
import sys
import os
import torch
import torch.amp
from pre_train.train_diffuser import TrainDiffuser
from models.diffuser_model import UNet
from torch_datasets.diffusion_dataset import DiffusionDataset
from torch.utils.data import DataLoader


if __name__ == "__main__":
    data_dir = os.path.join(os.getcwd(), "arc-all")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(image_channels=1)
    diffuser_train_dataset = DiffusionDataset(data_dir=data_dir, device=device)
    diffusion_dataloader = DataLoader(diffuser_train_dataset, batch_size=4, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    loss = torch.nn.MSELoss()
    scaler = torch.amp.grad_scaler.GradScaler()

    train_diffuser = TrainDiffuser(model, diffusion_dataloader, device, image_shape=(1, 30, 30), optimizer=optimizer, loss_fn=loss, scaler=scaler)
    train_diffuser.train_model()