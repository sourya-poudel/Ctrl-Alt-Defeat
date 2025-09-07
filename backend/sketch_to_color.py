import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

# -------------------------------
#   U-Net Generator
# -------------------------------
class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=64):
        super(UNetGenerator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features, features * 2, 4, 2, 1),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features * 2, features * 4, 4, 2, 1),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(features * 4, features * 2, 4, 2, 1),
            nn.BatchNorm2d(features * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(features * 2, features, 4, 2, 1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(features, out_channels, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# -------------------------------
#   Denormalize function
# -------------------------------
def denormalize(tensor):
    """ Convert from [-1, 1] to [0, 1] """
    return (tensor * 0.5 + 0.5).clamp(0, 1)

# -------------------------------
#   Load Generator
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = UNetGenerator().to(device)

# Load the converted checkpoint
checkpoint_path = "generator_pix2pix.pth"  # Replace if you renamed
generator.load_state_dict(torch.load(checkpoint_path, map_location=device))
generator.eval()

# -------------------------------
#   Image Transformations
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# -------------------------------
#   Function to generate colored image
# -------------------------------
def generate_image(sketch_path, output_path="images/colored/output.jpg"):
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    
    sketch = Image.open(sketch_path).convert("RGB")
    sketch_tensor = transform(sketch).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output_tensor = generator(sketch_tensor).squeeze(0).cpu()
    
    output_img = denormalize(output_tensor).permute(1, 2, 0).numpy()
    output_img = (output_img * 255).astype(np.uint8)
    
    # Save the result
    Image.fromarray(output_img).save(output_path)
    
    # Display input vs output
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(sketch)
    axes[0].set_title("Input Sketch")
    axes[0].axis("off")
    
    axes[1].imshow(output_img)
    axes[1].set_title("Generated Image")
    axes[1].axis("off")
    
    plt.show()

# -------------------------------
#   Example Usage
# -------------------------------
# generate_image("images/sketches/example_sketch.png")
