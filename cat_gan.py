import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm

#Configs and Hyperparameters
DATA_FOLDERS = ["./dataset-part1", "./dataset-part2", "./dataset-part3"]
WORK_DIR = "./gan_outputs"   
BATCH_SIZE = 128
IMAGE_SIZE = 64
CHANNELS = 3
LATENT_DIM = 100             # Size of the z latent vector
EPOCHS = 100                 
LR = 0.0002                  
BETA1 = 0.5                
SEED = 42                    # For reproducibility
NUM_WORKERS = 4             

random.seed(SEED)
torch.manual_seed(SEED)

os.makedirs(WORK_DIR, exist_ok=True)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    torch.backends.cudnn.benchmark = True


#Dataset (loads images from one or more folders, no class-subfolder needed)
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

class FlatImageDataset(Dataset):
    def __init__(self, folders, transform=None):
        self.transform = transform
        self.paths = []
        for folder in folders:
            for root, _dirs, files in os.walk(folder):
                for fname in files:
                    if fname.lower().endswith(IMG_EXTENSIONS):
                        self.paths.append(os.path.join(root, fname))
        if not self.paths:
            raise RuntimeError(f"No images found in {folders}")
        print(f"Found {len(self.paths)} images across {len(folders)} folder(s).")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, 0  # dummy label, unused by the GAN


#Model Definitions (DCGAN)
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input is Z, going into a convolution
            nn.ConvTranspose2d(LATENT_DIM, 64 * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, CHANNELS, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input is (3) x 64 x 64
            nn.Conv2d(CHANNELS, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64 * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


if __name__ == "__main__":
    dataset = FlatImageDataset(DATA_FOLDERS,
                               transform=transforms.Compose([
                                   transforms.Resize(IMAGE_SIZE),
                                   transforms.CenterCrop(IMAGE_SIZE),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                             shuffle=True, num_workers=NUM_WORKERS)

    # Instantiate models and apply weights
    netG = Generator().to(device)
    netG.apply(weights_init)

    netD = Discriminator().to(device)
    netD.apply(weights_init)

    
    #Loss & Optimizers
    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, LATENT_DIM, 1, 1, device=device)  # For consistent evaluation images
    real_label = 1.
    fake_label = 0.

    optimizerD = optim.Adam(netD.parameters(), lr=LR, betas=(BETA1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=LR, betas=(BETA1, 0.999))


    #Training Loop
    best_g_loss = float('inf')
    checkpoint_interval = max(1, EPOCHS // 4)  # Every 25% of epochs

    g_loss_history = []
    d_loss_history = []

    print("Starting Training")
    for epoch in range(1, EPOCHS + 1):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}")

        epoch_g_loss = 0.0
        epoch_d_loss = 0.0

        for i, data in enumerate(pbar):
            netD.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

            output = netD(real_cpu).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(b_size, LATENT_DIM, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)

            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()

            errD = errD_real + errD_fake
            optimizerD.step()

            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            epoch_g_loss += errG.item()
            epoch_d_loss += errD.item()

            pbar.set_postfix({
                'Loss_D': f"{errD.item():.4f}",
                'Loss_G': f"{errG.item():.4f}",
                'D(x)': f"{D_x:.4f}",
                'D(G(z))': f"{D_G_z1:.4f}/{D_G_z2:.4f}"
            })

        avg_g_loss = epoch_g_loss / len(dataloader)
        avg_d_loss = epoch_d_loss / len(dataloader)
        g_loss_history.append(avg_g_loss)
        d_loss_history.append(avg_d_loss)


        #Checkpoints (Every 25%)
        if epoch % checkpoint_interval == 0 or epoch == EPOCHS:
            g_checkpoint_path = os.path.join(WORK_DIR, f'netG_epoch_{epoch}.pth')
            d_checkpoint_path = os.path.join(WORK_DIR, f'netD_epoch_{epoch}.pth')
            torch.save(netG.state_dict(), g_checkpoint_path)
            torch.save(netD.state_dict(), d_checkpoint_path)

            #Also save optimizer state so training can be resumed later
            torch.save({
                'epoch': epoch,
                'optimizerG_state_dict': optimizerG.state_dict(),
                'optimizerD_state_dict': optimizerD.state_dict(),
            }, os.path.join(WORK_DIR, f'training_state_epoch_{epoch}.pth'))

            with torch.no_grad():
                fake_eval = netG(fixed_noise).detach().cpu()
            img_path = os.path.join(WORK_DIR, f'sample_epoch_{epoch}.png')
            vutils.save_image(fake_eval, img_path, normalize=True)
            print(f"\n[Checkpoint] Saved model and sample images to {WORK_DIR}")

    
        #Save the "Best" Model
        if avg_g_loss < best_g_loss:
            best_g_loss = avg_g_loss
            best_path = os.path.join(WORK_DIR, 'best_netG.pth')
            torch.save(netG.state_dict(), best_path)

    print(f"\nTraining Complete! Best generator model saved as 'best_netG.pth' with an average loss of {best_g_loss:.4f}.")

    #Plot loss curves
    plt.figure(figsize=(8, 5))
    plt.plot(g_loss_history, label="Generator")
    plt.plot(d_loss_history, label="Discriminator")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("GAN Training Loss")
    plt.legend()
    plot_path = os.path.join(WORK_DIR, "loss_curve.png")
    plt.savefig(plot_path)
    print(f"Loss curve saved to {plot_path}")