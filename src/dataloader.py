import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torch
from torchvision import transforms
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

class EGDDataset(Dataset):
    def __init__(self, root_dir, target_class=None, transform=None, use_test_data=False):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []
        self.target_class = target_class  

        
        # Load training data
        train_dir = os.path.join(root_dir, 'train')
        for label in os.listdir(train_dir):
            if self.target_class is not None and int(label) != self.target_class:
                continue  # Skip classes that are not the target class
            label_path = os.path.join(train_dir, label)
            for file_name in os.listdir(label_path):
                if file_name.endswith('.BMP'):
                    file_path = os.path.join(label_path, file_name)
                    self.data.append(file_path)
                    self.labels.append(int(label))
                    
        # Optionally load test data
        if use_test_data:
            test_dir = os.path.join(root_dir, 'test')
            for label in os.listdir(test_dir):
                if self.target_class is not None and int(label) != self.target_class:
                    continue
                label_path = os.path.join(test_dir, label)
                for file_name in os.listdir(label_path):
                    if file_name.endswith('.BMP'):
                        file_path = os.path.join(label_path, file_name)
                        self.data.append(file_path)
                        self.labels.append(int(label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Image loading
        img_path = self.data[idx]
        image = Image.open(img_path)

        # Label
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label
    
        
        
#Now to test the dataloader:
if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    dataset = EGDDataset(root_dir='/home/ndelafuente/CVC/VAE_GASTROSCOPY/data_egd', target_class=0, transform=transform, use_test_data=False)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=64, num_workers=4)
    images, labels = next(iter(dataloader))
    print(images.shape)
    print(labels.shape)
    