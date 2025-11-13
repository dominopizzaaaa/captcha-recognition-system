import os
import random
from glob import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils as vutils

# ---------------------------
# 1. Config
# ---------------------------

data_root = "data_letter"        # folder that contains "custom/"
out_dir = "outputs"       # where samples + models get saved
os.makedirs(out_dir, exist_ok=True)
os.makedirs(os.path.join(out_dir, "samples"), exist_ok=True)
os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)

image_size = 64           # 64x64 images
nc = 3                    # number of channels: 3 for RGB, 1 for grayscale
batch_size = 128
num_epochs = 50

z_dim = 100               # latent vector size
ngf = 64                  # generator feature maps
ndf = 64                  # discriminator feature maps

lr = 2e-4
beta1 = 0.5
beta2 = 0.999

manual_seed = 999
random.seed(manual_seed)
torch.manual_seed(manual_seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------------------
# 2. Dataset & DataLoader
# ---------------------------

# Transform: resize, crop, tensor, normalize to [-1, 1]
if nc == 3:
    norm_mean = [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]
else:
    norm_mean = [0.5]
    norm_std = [0.5]

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

# Expect structure: data/custom/*.jpg
dataset = datasets.ImageFolder(root=data_root, transform=transform)
# sanity check
print(f"Found {len(dataset)} images in {data_root}")

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

# ---------------------------
# 3. Model definitions (DCGAN)
# ---------------------------

def weights_init(m):
    """Init weights as in DCGAN paper."""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, z_dim=100, ngf=64, nc=3):
        super().__init__()
        self.main = nn.Sequential(
            # input: Z latent vector Z going into a convolution
            # (z_dim) x 1 x 1  -> (ngf*8) x 4 x 4
            nn.ConvTranspose2d(z_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            # (ngf*8) x 4 x 4 -> (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # (ngf*4) x 8 x 8 -> (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # (ngf*2) x 16 x 16 -> (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # (ngf) x 32 x 32 -> (nc) x 64 x 64
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),  # output in [-1, 1]
        )

    def forward(self, z):
        return self.main(z)


class Discriminator(nn.Module):
    def __init__(self, ndf=64, nc=3):
        super().__init__()
        self.main = nn.Sequential(
            # input: (nc) x 64 x 64 -> (ndf) x 32 x 32
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf) x 32 x 32 -> (ndf*2) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf*2) x 16 x 16 -> (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf*4) x 8 x 8 -> (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf*8) x 4 x 4 -> 1 x 1 x 1
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),  # probability
        )

    def forward(self, x):
        out = self.main(x)
        return out.view(-1, 1)

if __name__ == "__main__":

    # Instantiate and init
    G = Generator(z_dim=z_dim, ngf=ngf, nc=nc).to(device)
    D = Discriminator(ndf=ndf, nc=nc).to(device)

    G.apply(weights_init)
    D.apply(weights_init)

    print(G)
    print(D)

    # ---------------------------
    # 4. Loss & Optimisers
    # ---------------------------

    criterion = nn.BCELoss()

    optimizerD = optim.Adam(D.parameters(), lr=lr, betas=(beta1, beta2))
    optimizerG = optim.Adam(G.parameters(), lr=lr, betas=(beta1, beta2))

    # fixed noise for monitoring progression
    fixed_noise = torch.randn(64, z_dim, 1, 1, device=device)

    # ---------------------------
    # 5. Training loop
    # ---------------------------

    step = 0
    for epoch in range(num_epochs):
        for i, (real_images, _) in enumerate(dataloader):
            real_images = real_images.to(device)
            b_size = real_images.size(0)

            # Create labels
            real_label = torch.ones(b_size, 1, device=device)
            fake_label = torch.zeros(b_size, 1, device=device)

            # ----- Update D: maximize log(D(x)) + log(1 - D(G(z)))
            # Train with real
            D.zero_grad()
            output_real = D(real_images)
            lossD_real = criterion(output_real, real_label)

            # Train with fake
            noise = torch.randn(b_size, z_dim, 1, 1, device=device)
            fake_images = G(noise).detach()  # detach so G is not updated here
            output_fake = D(fake_images)
            lossD_fake = criterion(output_fake, fake_label)

            lossD = lossD_real + lossD_fake
            lossD.backward()
            optimizerD.step()

            # ----- Update G: maximize log(D(G(z)))  (i.e. fool D)
            G.zero_grad()
            noise = torch.randn(b_size, z_dim, 1, 1, device=device)
            fake_images = G(noise)
            output_fake_for_G = D(fake_images)
            lossG = criterion(output_fake_for_G, real_label)  # want D(fake) -> 1
            lossG.backward()
            optimizerG.step()

            if i % 50 == 0:
                print(
                    f"[Epoch {epoch+1}/{num_epochs}] "
                    f"[Batch {i+1}/{len(dataloader)}] "
                    f"Loss_D: {lossD.item():.4f} "
                    f"Loss_G: {lossG.item():.4f}"
                )

            # save sample grid occasionally
            if step % 500 == 0:
                with torch.no_grad():
                    fake = G(fixed_noise).detach().cpu()
                # denormalize from [-1,1] to [0,1]
                fake = fake * 0.5 + 0.5
                sample_path = os.path.join(out_dir, "samples", f"step_{step:06d}.png")
                vutils.save_image(fake, sample_path, nrow=8)
            step += 1

        # Save model checkpoints each epoch
        torch.save(G.state_dict(), os.path.join(out_dir, "checkpoints", f"G_epoch_{epoch+1}.pt"))
        torch.save(D.state_dict(), os.path.join(out_dir, "checkpoints", f"D_epoch_{epoch+1}.pt"))
        print(f"Saved checkpoints for epoch {epoch+1}")

    print("Training complete.")
