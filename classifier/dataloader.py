from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
import torch
from torchvision import transforms
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

class EGDDataset(Dataset):
    def __init__(self, root_dir, transform=None, use_test_data=False, only_test_data=False):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []

        # Load training data
        train_dir = os.path.join(root_dir, 'train')
        for label in os.listdir(train_dir):
            label_path = os.path.join(train_dir, label)
            for file_name in os.listdir(label_path):
                if file_name.endswith('.BMP') or file_name.endswith('.bmp'):
                    file_path = os.path.join(label_path, file_name)
                    self.data.append(file_path)
                    self.labels.append(int(label))

        # Optionally load test data
        if use_test_data:
            test_dir = os.path.join(root_dir, 'test')
            for label in os.listdir(test_dir):
                label_path = os.path.join(test_dir, label)
                for file_name in os.listdir(label_path):
                    if file_name.endswith('.BMP'):
                        file_path = os.path.join(label_path, file_name)
                        self.data.append(file_path)
                        self.labels.append(int(label))
                        
        # Optionally load only test data
        if only_test_data:
            #remove training data
            self.data = []
            self.labels = []
            
            #load test data
            test_dir = os.path.join(root_dir, 'test')
            for label in os.listdir(test_dir):
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

# Example usage
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((560, 640))
    ])

    # Create dataset for training
    train_dataset = EGDDataset(root_dir="/home/ndelafuente/CVC/EGD_Barcelona/gastroscopy_attention_classifier/TRAIN&MODEL_4_GENERATEDDATA/data", transform=transform, use_test_data=False)
    
    # Create dataset for testing
    test_dataset = EGDDataset(root_dir="/home/ndelafuente/CVC/EGD_Barcelona/gastroscopy_attention_classifier/TRAIN&MODEL_4_GENERATEDDATA/data", transform=transform, use_test_data=True, only_test_data=True)

    # Example of accessing the dataset
    print(train_dataset[0])  # Output will be (image, label) for the first image in the training set
    print(test_dataset[0])   # Output will be (image, label) for the first image in the testing set

    print(len(train_dataset))
    print(len(test_dataset))
    
    plt.imshow(train_dataset[0][0].permute(1, 2, 0))
    plt.axis('off')
    plt.show()