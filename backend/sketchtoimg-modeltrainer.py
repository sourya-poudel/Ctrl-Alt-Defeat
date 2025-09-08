
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, x):
        return self.model(x)
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.down1 = DownSample(3, 64)
        self.down2 = DownSample(64, 128)
        self.down3 = DownSample(128, 256)
        self.down4 = DownSample(256, 512)
        self.down5 = DownSample(512, 512)
        self.down6 = DownSample(512, 512)
        self.down7 = DownSample(512, 512)
        self.down8 = DownSample(512, 512)

        #upsample
        self.up1 = Upsample(512, 512)    
        self.up2 = Upsample(1024, 512)
        self.up3 = Upsample(1024, 512)
        self.up4 = Upsample(1024, 512)
        self.up5 = Upsample(1024, 256)
        self.up6 = Upsample(512, 128)
        self.up7 = Upsample(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, 3, 4, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8, d7)   
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        return self.final(u7)


class PairedImageDataset(Dataset):
    def __init__(self, sketch_dir, img_dir, transform=None):
        self.sketch_dir = sketch_dir
        self.img_dir = img_dir
        self.transform = transform
        sketch_files = set([f for f in os.listdir(sketch_dir) if os.path.isfile(os.path.join(sketch_dir, f))])
        img_files = set([f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))])
        self.filenames = sorted(list(sketch_files & img_files))
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        sketch_path = os.path.join(self.sketch_dir, fname)
        img_path = os.path.join(self.img_dir, fname)

        sketch = Image.open(sketch_path).convert("RGB")
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            sketch = self.transform(sketch)
            img = self.transform(img)
        return sketch, img

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sketch_dir = "images/Paired_Images-perfect/sketch"
    img_dir = "images/Paired_Images-perfect/real"
    os.makedirs("checkpoints", exist_ok=True)
    save_path = os.path.join("checkpoints", "generator_pix2pix.pth")

    batch_size = 4
    epochs = 50
    lr = 2e-4

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    dataset = PairedImageDataset(sketch_dir, img_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)

    print(f"Found {len(dataset)} paired images. Batch size {batch_size}. Steps/epoch: {len(dataloader)}")

    generator = Generator().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(1, epochs + 1):
        generator.train()
        running_loss = 0.0
        for i, (sketch, img) in enumerate(dataloader):
            sketch = sketch.to(device)
            img = img.to(device)

            optimizer.zero_grad()
            output = generator(sketch)
            loss = criterion(output, img)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 10 == 0 or (i + 1) == len(dataloader):
                print(f"[Epoch {epoch}/{epochs}] Step {i+1}/{len(dataloader)} - batch loss: {loss.item():.4f}")

        avg_loss = running_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        print(f"Epoch {epoch} completed. Avg Loss: {avg_loss:.4f}")
#save checkpoint
        if epoch % 10 == 0:
            torch.save(generator.state_dict(), save_path)
            print(f"Saved checkpoint at epoch {epoch} -> {save_path}")

    torch.save(generator.state_dict(), save_path)
    print("Training finished. Final model saved to:", save_path)

if __name__ == "__main__":
    train()

