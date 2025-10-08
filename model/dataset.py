import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import random
# Corrected import: 'transforms' module is now available
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

# ---- Paired Augmentation Helpers ----
class PairedCompose:
    def __init__(self, operations):
        self.operations = operations
    def __call__(self, img1, img2):
        for op in self.operations:
            img1, img2 = op(img1, img2)
        return img1, img2

class PairedRandomHorizontalFlip:
    def __init__(self, prob=0.5): self.prob = prob
    def __call__(self, img1, img2):
        if random.random() < self.prob:
            return F.hflip(img1), F.hflip(img2)
        return img1, img2

class PairedRandomVerticalFlip:
    def __init__(self, prob=0.2): self.prob = prob
    def __call__(self, img1, img2):
        if random.random() < self.prob:
            return F.vflip(img1), F.vflip(img2)
        return img1, img2

class PairedRandomRotation:
    def __init__(self, degrees=15): self.degrees = degrees
    def __call__(self, img1, img2):
        angle = random.uniform(-self.degrees, self.degrees)
        return F.rotate(img1, angle), F.rotate(img2, angle)

class PairedColorJitter:
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        self.tj = transforms.ColorJitter(brightness, contrast, saturation, hue)
    def __call__(self, img1, img2):
        state = torch.get_rng_state()
        img1 = self.tj(img1)
        torch.set_rng_state(state)
        img2 = self.tj(img2)
        return img1, img2

class PairedResize:
    def __init__(self, size=(256, 256)):
        self.size = size
        self.resize = transforms.Resize(self.size)
    def __call__(self, img1, img2):
        return self.resize(img1), self.resize(img2)

class PairedToTensor:
    def __call__(self, img1, img2):
        return F.to_tensor(img1), F.to_tensor(img2)

# ---- The Unified Dataset Class ----
class EuvpDataset(Dataset):
    def __init__(self, base_dirs, split='train', transform=None):
        self.input_paths = []
        self.target_paths = []
        self.transform = transform
        self.split = split

        if split == 'train':
            subfolder_A = 'trainA'
            subfolder_B = 'trainB'
        elif split == 'validation':
            subfolder_A = 'validation'
            subfolder_B = 'validation'
        elif split == 'test':
            subfolder_A = 'Inp'
            subfolder_B = 'GTr'
        else:
            raise ValueError(f"Invalid split: {split}")

        for base_dir in base_dirs:
            dirA = os.path.join(base_dir, subfolder_A)
            dirB = os.path.join(base_dir, subfolder_B)
            if not os.path.exists(dirA) or not os.path.exists(dirB):
                print(f"Warning: Directory not found. Skipping {base_dir}")
                continue
            
            filesA = sorted(os.listdir(dirA))
            filesB = sorted(os.listdir(dirB))
            
            file_pairs = [f for f in filesA if f in filesB]
            
            for f in file_pairs:
                self.input_paths.append(os.path.join(dirA, f))
                self.target_paths.append(os.path.join(dirB, f))
        
        print(f"Found {len(self.input_paths)} {split} image pairs.")

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        try:
            input_img = Image.open(self.input_paths[idx]).convert('RGB')
            target_img = Image.open(self.target_paths[idx]).convert('RGB')
        except Exception:
            return None
        
        if self.transform:
            input_img, target_img = self.transform(input_img, target_img)
        
        return input_img, target_img

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

def create_dataloaders(dataset_base_path, batch_size=4, num_workers=4):
    paired_domains = [
        os.path.join(dataset_base_path, "Paired", "underwater_dark"),
        os.path.join(dataset_base_path, "Paired", "underwater_imagenet"),
        os.path.join(dataset_base_path, "Paired", "underwater_scenes")
    ]
    test_base_dir = os.path.join(dataset_base_path, "test_samples")

    train_transform_ops = PairedCompose([
        PairedResize((256, 256)),
        PairedRandomHorizontalFlip(prob=0.5),
        PairedRandomVerticalFlip(prob=0.2),
        PairedRandomRotation(degrees=15),
        PairedColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        PairedToTensor(),
    ])

    val_test_transform_ops = PairedCompose([
        PairedResize((256, 256)),
        PairedToTensor(),
    ])

    train_dataset = EuvpDataset(paired_domains, split='train', transform=train_transform_ops)
    validation_dataset = EuvpDataset(paired_domains, split='validation', transform=val_test_transform_ops)
    test_dataset = EuvpDataset([test_base_dir], split='test', transform=val_test_transform_ops)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)

    return train_loader, validation_loader, test_loader