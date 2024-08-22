import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda import amp
import os
import gc
import itertools
from matplotlib import pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
import torchvision.transforms as transforms
from tqdm import tqdm
import json
from models.cycleGAN import *
from torch_datasets.gan_dataset import GanDataset
from torchvision.utils import make_grid, save_image
from labml import tracker, experiment, monit

class TrainGAN:
    def __init__(self):
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Hyper-parameters
        self.epochs = 20
        self.batch_size = 1
        self.dataloader_workers = 4
        self.learning_rate = 0.001
        self.adam_betas = (0.5, 0.999)
        self.decay_start = 5

        # The paper suggests using a least-squares loss instead of
        # negative log-likelihood, at it is found to be more stable.
        self.gan_loss = torch.nn.MSELoss()

        # L1 loss is used for cycle loss and identity loss
        self.cycle_loss = torch.nn.L1Loss()
        self.identity_loss = torch.nn.L1Loss()

        # Image dimensions
        self.img_height = 32
        self.img_width = 32
        self.img_channels = 1

        # Number of residual blocks in the generator
        self.n_residual_blocks = 15

        # Loss coefficients
        self.cyclic_loss_coefficient = 10.0
        self.identity_loss_coefficient = 5.

        self.sample_interval = 10000

        # Models
        self.generator_xy: GeneratorResNet
        self.generator_yx: GeneratorResNet
        self.discriminator_x: Discriminator
        self.discriminator_y: Discriminator

        # Optimizers
        self.generator_optimizer: torch.optim.Adam
        self.discriminator_optimizer: torch.optim.Adam

        # Learning rate schedules
        self.generator_lr_scheduler: torch.optim.lr_scheduler.LambdaLR
        self.discriminator_lr_scheduler: torch.optim.lr_scheduler.LambdaLR

        # Data loaders
        self.data_dir = "D:/Praetorian-ARC-AGI/arc-all"
        self.dataloader: DataLoader
        self.valid_dataloader: DataLoader

    def sample_images(self, n: int, output_dir: str = "./sample_images"):
        """Generate samples from test set and save them as color-mapped images"""
        os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

        # Define a color mapping for integers from 0 to 9 (example)
        color_map = {
            0: (0, 0, 0),       # Black
            1: (255, 0, 0),     # Red
            2: (0, 255, 0),     # Green
            3: (0, 0, 255),     # Blue
            4: (255, 255, 0),   # Yellow
            5: (255, 0, 255),   # Magenta
            6: (0, 255, 255),   # Cyan
            7: (255, 165, 0),   # Orange
            8: (128, 0, 128),   # Purple
            9: (255, 255, 255)  # White
        }

        batch = next(iter(self.valid_dataloader))
        self.generator_xy.eval()
        self.generator_yx.eval()
        with torch.no_grad():
            data_x, data_y = batch['x'].to(self.generator_xy.device), batch['y'].to(self.generator_yx.device)
            gen_y = self.generator_xy(data_x)
            gen_x = self.generator_yx(data_y)

            # Multiply by 9 and round to nearest integer
            data_x = torch.round(data_x * 9).cpu().numpy().astype(int)
            data_y = torch.round(data_y * 9).cpu().numpy().astype(int)
            gen_x = torch.round(gen_x * 9).cpu().numpy().astype(int)
            gen_y = torch.round(gen_y * 9).cpu().numpy().astype(int)

            # Convert the tensors to color images using the color map
            def map_to_color(tensor):
                # Ensure tensor is 2D by squeezing out any channel dimension
                if tensor.ndim == 3:
                    tensor = tensor.squeeze(0)  # Assuming tensor shape is (1, height, width)
                elif tensor.ndim == 4:
                    tensor = tensor.squeeze(0).squeeze(0)  # Assuming tensor shape is (batch_size, 1, height, width)

                height, width = tensor.shape  # Get height and width from the squeezed tensor
                color_image = np.zeros((height, width, 3), dtype=np.uint8)  # Initialize a blank RGB image

                # Apply color mapping
                for int_val, color in color_map.items():
                    mask = (tensor == int_val)  # Create a mask for the current integer value
                    color_image[mask] = color  # Apply the color where the mask is True

                return Image.fromarray(color_image)


            # Save the images
            map_to_color(data_x).save(os.path.join(output_dir, f"original_x_{n}.png"))
            map_to_color(gen_y).save(os.path.join(output_dir, f"generated_y_{n}.png"))
            map_to_color(data_y).save(os.path.join(output_dir, f"original_y_{n}.png"))
            map_to_color(gen_x).save(os.path.join(output_dir, f"generated_x_{n}.png"))

        print(f"Color-mapped images saved in {output_dir}")

    def initialize(self):
        """
        ## Initialize models and data loaders
        """
        input_shape = (self.img_channels, self.img_height, self.img_width)

        # Create the models
        self.generator_xy = GeneratorResNet(self.img_channels, self.n_residual_blocks).to(self.device)
        self.generator_yx = GeneratorResNet(self.img_channels, self.n_residual_blocks).to(self.device)
        self.discriminator_x = Discriminator(input_shape).to(self.device)
        self.discriminator_y = Discriminator(input_shape).to(self.device)

        # Create the optmizers
        self.generator_optimizer = torch.optim.Adam(
            itertools.chain(self.generator_xy.parameters(), self.generator_yx.parameters()),
            lr=self.learning_rate, betas=self.adam_betas)
        self.discriminator_optimizer = torch.optim.Adam(
            itertools.chain(self.discriminator_x.parameters(), self.discriminator_y.parameters()),
            lr=self.learning_rate, betas=self.adam_betas)

        # Create the learning rate schedules.
        # The learning rate stars flat until `decay_start` epochs,
        # and then linearly reduce to $0$ at end of training.
        decay_epochs = self.epochs - self.decay_start
        self.generator_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.generator_optimizer, lr_lambda=lambda e: 1.0 - max(0, e - self.decay_start) / decay_epochs)
        self.discriminator_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.discriminator_optimizer, lr_lambda=lambda e: 1.0 - max(0, e - self.decay_start) / decay_epochs)

        # Training data loader
        self.dataloader = DataLoader(
            GanDataset(self.data_dir, self.device, "train"),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.dataloader_workers,
        )

        # Validation data loader
        self.valid_dataloader = DataLoader(
            GanDataset(self.data_dir, self.device, "val"),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.dataloader_workers,
        )

    def run(self):
        
        # Replay buffers to keep generated samples
        gen_x_buffer = ReplayBuffer()
        gen_y_buffer = ReplayBuffer()
        best_ssim = 0
        best_mse = 1
        # Loop through epochs
        for epoch in monit.loop(self.epochs):
            # Loop through the dataset
            for i, batch in monit.enum('Train', self.dataloader):
                # Move images to the device
                data_x, data_y = batch['x'].to(self.device), batch['y'].to(self.device)

                # true labels equal to $1$
                true_labels = torch.ones(data_x.size(0), *self.discriminator_x.output_shape,
                                         device=self.device, requires_grad=False)
                # false labels equal to $0$
                false_labels = torch.zeros(data_x.size(0), *self.discriminator_x.output_shape,
                                           device=self.device, requires_grad=False)

                # Train the generators.
                # This returns the generated images.
                gen_x, gen_y = self.optimize_generators(data_x, data_y, true_labels)

                #  Train discriminators
                self.optimize_discriminator(data_x, data_y,
                                            gen_x_buffer.push_and_pop(gen_x), gen_y_buffer.push_and_pop(gen_y),
                                            true_labels, false_labels)

                # Save training statistics and increment the global step counter
                tracker.save()
                tracker.add_global_step(max(len(data_x), len(data_y)))

                # Save images at intervals
                batches_done = epoch * len(self.dataloader) + i
                if batches_done % self.sample_interval == 0:
                    # Sample images
                    self.sample_images(batches_done)

            # Save the best model when epoch finishes
            avg_mse, avg_ssim = self.evaluate()   
            if avg_mse < best_mse or avg_ssim > best_ssim: 
                # Save generator_xy model state_dict
                checkpoint_path = os.path.join(f"{os.getcwd()}/checkpoints/cyclegan_checkpoints", f"generator_xy_checkpoint_epoch_{epoch}_run_2.pth")
                torch.save(self.generator_xy.state_dict(), checkpoint_path)
                print(f"Saved best model to: {checkpoint_path}")
                best_mse = avg_mse
                best_ssim = avg_ssim

            # Update learning rates
            self.generator_lr_scheduler.step()
            self.discriminator_lr_scheduler.step()
            # New line
            tracker.new_line()

    def optimize_generators(self, data_x: torch.Tensor, data_y: torch.Tensor, true_labels: torch.Tensor):
        """
        ### Optimize the generators with identity, gan and cycle losses.
        """

        #  Change to training mode
        self.generator_xy.train()
        self.generator_yx.train()

        # Identity loss
        # $$\lVert F(G(x^{(i)})) - x^{(i)} \lVert_1\
        #   \lVert G(F(y^{(i)})) - y^{(i)} \rVert_1$$
        loss_identity = (self.identity_loss(self.generator_yx(data_x), data_x) +
                         self.identity_loss(self.generator_xy(data_y), data_y))

        # Generate images $G(x)$ and $F(y)$
        gen_y = self.generator_xy(data_x)
        gen_x = self.generator_yx(data_y)

        # GAN loss
        # $$\bigg(D_Y\Big(G\Big(x^{(i)}\Big)\Big) - 1\bigg)^2
        #  + \bigg(D_X\Big(F\Big(y^{(i)}\Big)\Big) - 1\bigg)^2$$
        loss_gan = (self.gan_loss(self.discriminator_y(gen_y), true_labels) +
                    self.gan_loss(self.discriminator_x(gen_x), true_labels))

        # Cycle loss
        # $$
        # \lVert F(G(x^{(i)})) - x^{(i)} \lVert_1 +
        # \lVert G(F(y^{(i)})) - y^{(i)} \rVert_1
        # $$
        loss_cycle = (self.cycle_loss(self.generator_yx(gen_y), data_x) +
                      self.cycle_loss(self.generator_xy(gen_x), data_y))

        # Total loss
        loss_generator = (loss_gan +
                          self.cyclic_loss_coefficient * loss_cycle +
                          self.identity_loss_coefficient * loss_identity)

        # Take a step in the optimizer
        self.generator_optimizer.zero_grad()
        loss_generator.backward()
        self.generator_optimizer.step()

        # Log losses
        tracker.add({'loss.generator': loss_generator,
                     'loss.generator.cycle': loss_cycle,
                     'loss.generator.gan': loss_gan,
                     'loss.generator.identity': loss_identity})

        # Return generated images
        return gen_x, gen_y

    def optimize_discriminator(self, data_x: torch.Tensor, data_y: torch.Tensor,
                               gen_x: torch.Tensor, gen_y: torch.Tensor,
                               true_labels: torch.Tensor, false_labels: torch.Tensor):
        """
        ### Optimize the discriminators with gan loss.
        """

        # GAN Loss
        #
        # \begin{align}
        # \bigg(D_Y\Big(y ^ {(i)}\Big) - 1\bigg) ^ 2
        # + D_Y\Big(G\Big(x ^ {(i)}\Big)\Big) ^ 2 + \\
        # \bigg(D_X\Big(x ^ {(i)}\Big) - 1\bigg) ^ 2
        # + D_X\Big(F\Big(y ^ {(i)}\Big)\Big) ^ 2
        # \end{align}
        loss_discriminator = (self.gan_loss(self.discriminator_x(data_x), true_labels) +
                              self.gan_loss(self.discriminator_x(gen_x), false_labels) +
                              self.gan_loss(self.discriminator_y(data_y), true_labels) +
                              self.gan_loss(self.discriminator_y(gen_y), false_labels))

        # Take a step in the optimizer
        self.discriminator_optimizer.zero_grad()
        loss_discriminator.backward()
        self.discriminator_optimizer.step()

        # Log losses
        tracker.add({'loss.discriminator': loss_discriminator})

    def evaluate(self):

        self.generator_xy.eval()
        total_mse = 0
        total_ssim = 0
        num_batches = len(self.valid_dataloader)
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.valid_dataloader, desc="Evaluating")):
                data_x = batch['x'].to(self.device)
                data_y = batch['y'].to(self.device)
                
                # Generate images
                gen_y = self.generator_xy(data_x)

                # Calculate MSE for this batch
                mse_batch = torch.nn.functional.mse_loss(gen_y, data_y).item()
                total_mse += mse_batch

                # Calculate SSIM for this batch
                # Note: SSIM expects the inputs to be in the range [0, 1] or [0, 255], so normalize back if needed
                ssim_batch = self.calculate_ssim(gen_y, data_y)
                total_ssim += ssim_batch

        # Compute the average metrics
        avg_mse = total_mse / num_batches
        avg_ssim = total_ssim / num_batches

        print(f"Average MSE: {avg_mse}")
        print(f"Average SSIM: {avg_ssim}")

        return avg_mse, avg_ssim

    def calculate_ssim(self, gen_y, data_y):
        """
        Calculate SSIM between generated and ground truth images
        """
        ssim_values = []
        for i in range(gen_y.size(0)):
            gen_img = gen_y[i].squeeze().cpu().numpy()
            true_img = data_y[i].squeeze().cpu().numpy()
            ssim_value = ssim(gen_img, true_img, data_range=gen_img.max() - gen_img.min())
            ssim_values.append(ssim_value)
        return np.mean(ssim_values)

    def display_comparison(self, data_x, data_y, gen_y):
        """
        Display a comparison of input, generated, and ground truth images
        """
        # Choose the first image in the batch for display
        data_x = data_x[0].cpu().squeeze().numpy()
        data_y = data_y[0].cpu().squeeze().numpy()
        gen_y = gen_y[0].cpu().squeeze().numpy()

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(data_x, cmap='gray')
        axs[0].set_title("Input Image (X)")
        axs[0].axis('off')

        axs[1].imshow(data_y, cmap='gray')
        axs[1].set_title("Ground Truth (Y)")
        axs[1].axis('off')

        axs[2].imshow(gen_y, cmap='gray')
        axs[2].set_title("Generated Image (G(X))")
        axs[2].axis('off')

        plt.show()