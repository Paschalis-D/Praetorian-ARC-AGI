import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import tqdm

class UNet(nn.Module):
    def __init__(self, in_channels=1, base_num_filters=64, embed_size=1, num_timesteps=1000):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.base_num_filters = base_num_filters
        self.embed_size = embed_size

        # Time embedding module
        self.time_embedding = nn.Sequential(
            nn.Linear(1, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, embed_size),
        )

        # Conv layers to expand time embedding channels to match each encoder/decoder stage
        self.time_emb_enc1 = nn.Conv2d(embed_size, base_num_filters, kernel_size=1)
        self.time_emb_enc2 = nn.Conv2d(embed_size, base_num_filters * 2, kernel_size=1)
        self.time_emb_enc3 = nn.Conv2d(embed_size, base_num_filters * 4, kernel_size=1)
        self.time_emb_enc4 = nn.Conv2d(embed_size, base_num_filters * 8, kernel_size=1)
        self.time_emb_bottleneck = nn.Conv2d(embed_size, base_num_filters * 16, kernel_size=1)
        self.time_emb_dec4 = nn.Conv2d(embed_size, base_num_filters * 8, kernel_size=1)
        self.time_emb_dec3 = nn.Conv2d(embed_size, base_num_filters * 4, kernel_size=1)
        self.time_emb_dec2 = nn.Conv2d(embed_size, base_num_filters * 2, kernel_size=1)
        self.time_emb_dec1 = nn.Conv2d(embed_size, base_num_filters, kernel_size=1)

        # Encoder path
        self.enc1 = self.encoder_block(in_channels, base_num_filters)
        self.enc2 = self.encoder_block(base_num_filters, base_num_filters * 2)
        self.enc3 = self.encoder_block(base_num_filters * 2, base_num_filters * 4)
        self.enc4 = self.encoder_block(base_num_filters * 4, base_num_filters * 8)

        # Bottleneck with additional residual block
        self.bottleneck = self.residual_block(base_num_filters * 8, base_num_filters * 16)

        # Decoder path
        self.dec4 = self.decoder_block(base_num_filters * 16, base_num_filters * 8)
        self.dec3 = self.decoder_block(base_num_filters * 8, base_num_filters * 4)
        self.dec2 = self.decoder_block(base_num_filters * 4, base_num_filters * 2)
        self.dec1 = self.decoder_block(base_num_filters * 2, base_num_filters)

        # Final convolution to reduce to the original number of channels
        self.final_conv = nn.Conv2d(base_num_filters, in_channels, kernel_size=1)

        # Self-Attention layers (for specific feature map resolutions)
        self.attn2 = nn.MultiheadAttention(base_num_filters * 4, num_heads=8, batch_first=True)
        self.attn3 = nn.MultiheadAttention(base_num_filters * 8, num_heads=8, batch_first=True)

    def forward(self, x, t):
        # Generate time embedding
        t = t.unsqueeze(-1).float()
        time_emb = self.time_embedding(t).unsqueeze(-1).unsqueeze(-1)

        # Encoder path with time embeddings
        enc1 = self.enc1(x + self.time_emb_enc1(time_emb.expand(-1, -1, x.size(2), x.size(3))))
        enc2 = self.enc2(enc1 + self.time_emb_enc2(time_emb.expand(-1, -1, enc1.size(2), enc1.size(3))))
        enc3 = self.enc3(enc2 + self.time_emb_enc3(time_emb.expand(-1, -1, enc2.size(2), enc2.size(3))))
        enc4 = self.enc4(enc3 + self.time_emb_enc4(time_emb.expand(-1, -1, enc3.size(2), enc3.size(3))))

        # Bottleneck
        bottleneck = self.bottleneck(enc4 + self.time_emb_bottleneck(time_emb.expand(-1, -1, enc4.size(2), enc4.size(3))))

        # Decoder path with skip connections and time embeddings
        dec4 = self.dec4(bottleneck + enc4 + self.time_emb_dec4(time_emb.expand(-1, -1, bottleneck.size(2), bottleneck.size(3))))
        dec3 = self.dec3(dec4 + enc3 + self.time_emb_dec3(time_emb.expand(-1, -1, dec4.size(2), dec4.size(3))))

        # Apply self-attention at the appropriate levels (16x16 and 8x8)
        dec3_flat = dec3.flatten(2).permute(0, 2, 1)
        dec3, _ = self.attn3(dec3_flat, dec3_flat, dec3_flat)
        dec3 = dec3.permute(0, 2, 1).view_as(dec4)

        dec2 = self.dec2(dec3 + enc2 + self.time_emb_dec2(time_emb.expand(-1, -1, dec3.size(2), dec3.size(3))))

        dec2_flat = dec2.flatten(2).permute(0, 2, 1)
        dec2, _ = self.attn2(dec2_flat, dec2_flat, dec2_flat)
        dec2 = dec2.permute(0, 2, 1).view_as(dec3)

        dec1 = self.dec1(dec2 + enc1 + self.time_emb_dec1(time_emb.expand(-1, -1, dec2.size(2), dec2.size(3))))

        # Final output
        out = self.final_conv(dec1)
        return out

    def residual_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
    def encoder_block(self, in_channels, num_filters):
        return nn.Sequential(
            self.residual_block(in_channels, num_filters),
            self.residual_block(num_filters, num_filters),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
    def decoder_block(self, in_channels, num_filters):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, num_filters, kernel_size=2, stride=2),
            self.residual_block(num_filters, num_filters)
        )




class SimpleDiffusion(nn.Module):
    def __init__(self, model, num_diffusion_timesteps = 1000, img_shape = (3, 32, 32), device = "cuda"):
        super(SimpleDiffusion, self).__init__()
        self.timesteps = num_diffusion_timesteps
        self.img_shape = img_shape
        self.device = device
        self.model = model
        self.initialize()

    
    def initialize(self):
        self.beta = self.get_betas()
        self.alpha = 1 - self.beta
        self.sqrt_beta = torch.sqrt(self.beta)
        self.alpha_cumulative = torch.cumprod(self.alpha, dim=0)
        self.sqrt_alpha_cumulative = torch.sqrt(self.alpha_cumulative)
        self.one_by_sqrt_alpha = 1. / torch.sqrt(self.alpha)
        self.sqrt_one_minus_alpha_cumulative = torch.sqrt(1 - self.alpha_cumulative)

    def get_betas(self):
        scale = 1000 / self.timesteps
        beta_start = scale * 1e-4
        beta_end = scale * 0.02
        return torch.linspace(
            beta_start,
            beta_end,
            self.timesteps,
            device=self.device,
            dtype=torch.float32
        )

    def forward_diffusion(self, x0, timesteps):
        eps = torch.randn_like(x0)  # Noise
        mean = self.get(self.sqrt_alpha_cumulative, t=timesteps) * x0
        std_dev = self.get(self.sqrt_one_minus_alpha_cumulative, t=timesteps)
        sample = mean + std_dev * eps
        return sample, eps

    @torch.no_grad()
    def reverse_diffusion(self, x, num_images, nrow, **kwargs):
        # x should now be the input image (not random noise)
        self.model.eval()

        for time_step in tqdm(iterable=reversed(range(1, self.timesteps)),
                            total=self.timesteps-1, dynamic_ncols=False,
                            desc="Sampling :: ", position=0):

            ts = torch.ones(num_images, dtype=torch.long, device=self.device) * time_step
            z = torch.randn_like(x) if time_step > 1 else torch.zeros_like(x)

            predicted_noise = self.model(x, ts)

            beta_t = self.get(self.beta, ts)
            one_by_sqrt_alpha_t = self.get(self.one_by_sqrt_alpha, ts)
            sqrt_one_minus_alpha_cumulative_t = self.get(self.sqrt_one_minus_alpha_cumulative, ts)

            x = (
                one_by_sqrt_alpha_t
                * (x - (beta_t / sqrt_one_minus_alpha_cumulative_t) * predicted_noise)
                + torch.sqrt(beta_t) * z
            )
        return x


    def get(self, array, t):
        return array[t].unsqueeze(1).unsqueeze(2).unsqueeze(3)


if __name__ == '__main__':
    pass