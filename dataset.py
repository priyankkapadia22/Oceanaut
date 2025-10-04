import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, ColorJitter
import torch
import random

# Custom transform to apply the same augmentation on image pairs
class PairedTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, img1, img2):
        # Seed to ensure same transform on both images
        seed = random.randint(0, 2 ** 32)
        random.seed(seed)
        torch.manual_seed(seed)
        img1 = self.base_transform(img1)
        random.seed(seed)
        torch.manual_seed(seed)
        img2 = self.base_transform(img2)
        return img1, img2

class EuvpDataset(Dataset):
    def __init__(self, base_dirs, split='train', transform=None):
        self.input_paths = []
        self.target_paths = []
        self.transform = transform
        
        if split == 'train':
            input_subfolder = 'trainA'
            target_subfolder = 'trainB'
        elif split == 'validation':
            input_subfolder = 'validation'
            target_subfolder = 'validation'
        elif split == 'test':
            input_subfolder = 'Inp'
            target_subfolder = 'GTr'
        else:
            raise ValueError("Invalid split. Choose from 'train', 'validation', or 'test'.")

        for base_dir in base_dirs:
            input_dir = os.path.join(base_dir, input_subfolder)
            target_dir = os.path.join(base_dir, target_subfolder)
            if not os.path.exists(input_dir) or not os.path.exists(target_dir):
                print(f"Warning: Directory not found. Skipping {base_dir}")
                continue
            try:
                file_names = sorted(os.listdir(input_dir))
                for file_name in file_names:
                    self.input_paths.append(os.path.join(input_dir, file_name))
                    self.target_paths.append(os.path.join(target_dir, file_name))
            except FileNotFoundError:
                print(f"Error: Could not list files in {input_dir}. Skipping.")
                continue
        print(f"Found {len(self.input_paths)} images for {split} split.")

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        try:
            input_image_path = self.input_paths[idx]
            target_image_path = self.target_paths[idx]
            input_image = Image.open(input_image_path).convert('RGB')
            target_image = Image.open(target_image_path).convert('RGB')
        except (IOError, ValueError):
            return None

        if self.transform:
            input_image, target_image = self.transform(input_image, target_image)

        return input_image, target_image

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

def create_dataloaders(dataset_base_path, batch_size=16):
    train_val_base_dirs = [
        os.path.join(dataset_base_path, 'Paired', 'underwater_dark'),
        os.path.join(dataset_base_path, 'Paired', 'underwater_imagenet'),
        os.path.join(dataset_base_path, 'Paired', 'underwater_scenes')
    ]
    test_base_dirs = [
        os.path.join(dataset_base_path, 'test_samples')
    ]

    base_transform = Compose([
        Resize((256, 256)),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.2),
        RandomRotation(degrees=15),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ToTensor(),
    ])
    paired_transform = PairedTransform(base_transform)

    train_dataset = EuvpDataset(base_dirs=train_val_base_dirs, split='train', transform=paired_transform)
    validation_dataset = EuvpDataset(base_dirs=train_val_base_dirs, split='validation', transform=paired_transform)
    test_dataset = EuvpDataset(base_dirs=test_base_dirs, split='test', transform=paired_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, validation_loader, test_loader