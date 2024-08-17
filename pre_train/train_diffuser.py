import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda import amp
import os
import gc
from tqdm import tqdm
from torchmetrics import MeanMetric
from models.diffuser_model import SimpleDiffusion

class TrainDiffuser:
    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        image_shape: tuple,
        optimizer: optim.Optimizer = optim.Adam,
        scaler: amp.GradScaler = torch.amp.GradScaler(),
        loss_fn: nn.Module = nn.MSELoss(),
        num_epochs: int = 100,
        timesteps: int = 1000,
        checkpoint_dir: str = os.getcwd()
    ):
        self.model = model
        self.loader = dataloader
        self.opt = optimizer
        self.scaler = scaler
        self.loss_fn = loss_fn
        self.epochs = num_epochs
        self.timesteps = timesteps
        self.device = device
        self.image_shape = image_shape
        self.sd = SimpleDiffusion(model=self.model, num_diffusion_timesteps=self.timesteps, img_shape=self.image_shape, device=self.device)
        self.checkpoint_dir = checkpoint_dir

    def train_one_epoch(self, epoch):
        loss_record = MeanMetric()
        self.model.train()

        with tqdm(total=len(self.loader), dynamic_ncols=True) as tq:
            tq.set_description(f"Train :: Epoch: {epoch}/{self.epochs}")

            for x0s, ys in self.loader:  # `ys` should be your target outputs
                tq.update(1)

                ts = torch.randint(low=1, high=self.timesteps, size=(x0s.shape[0],), device=self.device)
                xts, gt_noise = self.sd.forward_diffusion(x0s, ts)

                if torch.cuda.is_available():
                    with torch.amp.autocast(device_type='cuda'):
                        pred_noise = self.model(xts, ts)
                        loss = self.loss_fn(gt_noise, pred_noise)  # Loss for predicting noise

                        # Reverse diffusion to generate the final output
                        x_pred = self.sd.reverse_diffusion(xts, num_images=x0s.shape[0], nrow=1)

                        # Compare the generated image `x_pred` with the target `ys`
                        image_loss = self.loss_fn(x_pred, ys)  # New image-level loss

                        # Combine both losses (you can adjust the weighting as necessary)
                        total_loss = loss + image_loss
                else:
                    # Fallback to standard precision on CPU
                    pred_noise = self.model(xts, ts)
                    loss = self.loss_fn(gt_noise, pred_noise)  # Loss for predicting noise

                    # Reverse diffusion to generate the final output
                    x_pred = self.sd.reverse_diffusion(xts, num_images=x0s.shape[0], nrow=1)

                    # Compare the generated image `x_pred` with the target `ys`
                    image_loss = self.loss_fn(x_pred, ys)  # New image-level loss

                    # Combine both losses (you can adjust the weighting as necessary)
                    total_loss = loss + image_loss

                self.opt.zero_grad(set_to_none=True)
                self.scaler.scale(total_loss).backward()

                self.scaler.step(self.opt)
                self.scaler.update()

                loss_value = total_loss.detach().item()
                loss_record.update(loss_value)

                tq.set_postfix_str(s=f"Loss: {loss_value:.4f}")

            mean_loss = loss_record.compute().item()
            tq.set_postfix_str(s=f"Epoch Loss: {mean_loss:.4f}")

        return mean_loss
    
    def train_model(self):
        for epoch in range(1, self.epochs + 1):
            torch.cuda.empty_cache()
            gc.collect()
            
            self.train_one_epoch(epoch=epoch)
        
            if epoch % 20 == 0:
                save_path = os.path.join(self.checkpoint_dir, f"ckpt_epoch_{epoch}.pt")
                
                # Sampling (Algorithm 2)
                self.sd.reverse_diffusion(self.model, num_images=1, nrow=4)
        
                checkpoint_dict = {
                    "opt": self.opt.state_dict(),
                    "scaler": self.scaler.state_dict(),
                    "model": self.model.state_dict()
                }
                torch.save(checkpoint_dict, save_path)
                del checkpoint_dict

