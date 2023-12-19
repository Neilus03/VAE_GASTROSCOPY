'''
In this file, we will define all the configuration variables,
all the hyperparameters, and all the constants that we will use.
'''

'''
necessary imports for the config file
'''
import torch
from torchvision import transforms
from torch import optim
from torch.utils.data import DataLoader
from Conv_VAE_attention_model import ConvVAE
from dataloader import EGDDataset
import wandb

'''
Constants and other variables
'''
# Path to the Dataset folder
DATASET_PATH = '../data_egd'

# Path to the folder where we will save the model
MODELS_PATH = './models'

# Wandb project name and entity
WANDB_PROJECT_NAME = 'vae_gastroscopy'
ENTITY = 'neildlf'

# Saved images path for class 0
SAVED_IMAGES_PATH_0 = './saved_images_0'

# Saved images path for class 1
SAVED_IMAGES_PATH_1 = './saved_images_1'

# Saved images path for class 2
SAVED_IMAGES_PATH_2 = './saved_images_2'

#Generate images of class:
TARGET_CLASS = 2

'''
Hyperparameters
'''
# Size of the latent space
LATENT_SPACE_DIM = 256

# Model we will use for training
MODEL = ConvVAE(LATENT_SPACE_DIM)

# Number of epochs to train the model
NUM_EPOCHS = 1000

# Learning rate for the optimizer
LEARNING_RATE = 0.0001

# Optimizer to use for training
OPTIMIZER = optim.Adam(MODEL.parameters(), lr=LEARNING_RATE)

# Device to use for training
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Batch size for training
BATCH_SIZE = 4

# Number of workers for the dataloader
NUM_WORKERS = 4

# Number of channels in the input image
NUM_CHANNELS = 3

# Size of the image after resizing
IMAGE_SIZE = 224

# Transformations to apply to the images
TRANSFORMS = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

#Define the dataset
DATASET = EGDDataset(root_dir=DATASET_PATH, target_class=TARGET_CLASS, transform=TRANSFORMS, use_test_data=False)
# Dataloader for the EGD dataset
DATALOADER = DataLoader(DATASET, shuffle=True, batch_size = BATCH_SIZE,  num_workers=NUM_WORKERS)





