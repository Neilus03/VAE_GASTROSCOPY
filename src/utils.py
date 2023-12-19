import matplotlib.pyplot as plt
import wandb
import torch
import torch.nn.functional as F


def log_images(original, label, reconstructed):
    '''
    Description:
        Logs the original and reconstructed images to wandb for visualization
    Args:
        original: Original image
        label: Label of the original image
        reconstructed: Reconstructed image
    Returns:
        None. Images are logged to wandb
    '''
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    # Original image
    axs[0].imshow(original.permute(1, 2, 0).cpu().detach().numpy(), cmap='gray')
    axs[0].set_title(f'Original Image, Label: {label}')
    # Reconstructed image
    axs[1].imshow(reconstructed.permute(1, 2, 0).cpu().detach().numpy(), cmap='gray')
    axs[1].set_title('Reconstructed Image')
    plt.axis('off')
    wandb.log({"Original vs reconstructed image": plt})
    plt.close() # Prevent plots from accumulating in memory



def loss_function(recon_x, x, mu, logvar):
    '''
    Description:
        Computes the VAE loss function. 
        Loss will be the sum of the reconstruction loss and the KL divergence loss
        The reconstruction loss will be Mean Squared Error (MSE) loss
        The KL divergence loss will be the KL divergence between the learned mean and variance and the prior Gaussian distribution
        of the same dimensionality as the latent space
    Args:
        recon_x: Reconstructed image
        x: Original image
        mu: Learned mean of the latent space
        logvar: Learned log variance of the latent space
    Returns:
        The VAE loss, which is the sum of the reconstruction loss and the KL divergence loss
    '''
    # Normalized MSE loss
    recon_loss = F.mse_loss(recon_x.view(-1, 3*224*224), x.view(-1, 3*224*224), reduction='sum') / x.size(0)
    
    # KL divergence loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    
    return recon_loss + KLD