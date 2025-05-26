import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
import torch

class PairedDataset(Dataset):
    def __init__(self, low_dir, high_dir, transform=None, crop_size=None, training=True):
        self.low_dir = low_dir
        self.high_dir = high_dir
        self.transform = transform
        self.crop_size = crop_size
        self.training = training

        self.low_images = sorted([f for f in os.listdir(low_dir) if os.path.isfile(os.path.join(low_dir, f))])
        self.high_images = sorted([f for f in os.listdir(high_dir) if os.path.isfile(os.path.join(high_dir, f))])

        assert len(self.low_images) == len(self.high_images), "Mismatch in number of images"

    def __len__(self):
        return len(self.low_images)

    def __getitem__(self, idx):
        low_image_path = os.path.join(self.low_dir, self.low_images[idx])
        high_image_path = os.path.join(self.high_dir, self.high_images[idx])

        low_image = Image.open(low_image_path).convert('RGB')
        high_image = Image.open(high_image_path).convert('RGB')

        if self.transform:
            low_image = self.transform(low_image)
            high_image = self.transform(high_image)
        if self.training:
            if random.random() > 0.5:
                low_image = low_image.flip(-1) 
                high_image = high_image.flip(-1)
            if random.random() > 0.5:
                low_image = low_image.flip(-2)   
                high_image = high_image.flip(-2)
            k = random.randint(0, 3)
            low_image = torch.rot90(low_image, k, [1, 2])
            high_image = torch.rot90(high_image, k, [1, 2])

        if self.training and self.crop_size:
            i, j, h, w = transforms.RandomCrop.get_params(low_image, output_size=(self.crop_size, self.crop_size))
            low_image = transforms.functional.crop(low_image, i, j, h, w)
            high_image = transforms.functional.crop(high_image, i, j, h, w)

        return low_image, high_image

def create_dataloaders(train_low, train_high, test_low, test_high, crop_size=256, batch_size=1):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_loader = None
    test_loader = None

    if train_low and train_high:
        train_dataset = PairedDataset(train_low, train_high, transform=transform, crop_size=crop_size, training=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    if test_low and test_high:
        test_dataset = PairedDataset(test_low, test_high, transform=transform, training=False)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)

    return train_loader, test_loader
