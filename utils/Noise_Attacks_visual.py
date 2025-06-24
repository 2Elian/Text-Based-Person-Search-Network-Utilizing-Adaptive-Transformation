import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from PIL import Image
from .compute import compute_x_std


class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.bmp')]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        
        return image

# Define your transformations
transform = transforms.Compose([
    transforms.Resize((384, 128)),  # Resize the images to the desired size
    transforms.ToTensor(),  # Convert the images to Tensor
    transforms.Normalize(mean=[0.38901278, 0.3651612, 0.34836376], std=[0.24344306, 0.23738699, 0.23368555])  # Normalize the images
])

# Path to the image directory
image_dir = '/home/202312150002/my_paper/utils/images'

# Create the dataset
dataset = CustomImageDataset(image_dir=image_dir, transform=transform)

# Create the DataLoader
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

# Example: Iterate through the DataLoader
for images in dataloader:
    images = images.permute(0, 2, 3, 1) #[b,384,128,3]
    images = images.reshape(16, 384 * 128, 3) #[b,384*128,3]
    mu = torch.mean(images,dim=1,keepdim=True)
    sigma = compute_x_std(images)
    