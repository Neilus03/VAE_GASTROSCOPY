'''
File to train the VAE model on the EGD dataset (only train data)
'''
import torch
from utils import log_images, loss_function
import config
import wandb


#Initialize wandb
wandb.init(project=config.WANDB_PROJECT_NAME, entity=config.ENTITY, config={"learning_rate": config.LEARNING_RATE, "architecture": "ConvVAE", "dataset": "EGD", "batch_size": config.BATCH_SIZE})


# Training loop
for epoch in range(config.NUM_EPOCHS):
    config.MODEL.to(config.DEVICE).train()
    train_loss = 0

    for batch_idx, (data, label) in enumerate(config.DATALOADER):
        # Move data to device
        data = data.to(config.DEVICE)
        config.OPTIMIZER.zero_grad()

        # Forward pass through the model
        recon_batch, mu, logvar = config.MODEL(data)
        
        # Compute loss
        loss = loss_function(recon_batch, data, mu, logvar)
        train_loss += loss.item()

        # Backward pass and optimize
        loss.backward()
        config.OPTIMIZER.step()

        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()/len(data)}')
            # Visualize the first image in the batch
            
            label = label[0] # Get the label for the first image in the batch which is what we will visualize
            log_images(data[0], label.item(), recon_batch[0])
            # Log the loss
            wandb.log({"loss": loss.item()/len(data)})
            
    # Print average loss for the epoch
    print(f'====> Epoch: {epoch}, Average loss: {train_loss / len(config.DATALOADER.dataset)}')
    wandb.log({"average_loss": train_loss / len(config.DATALOADER.dataset)})
    # Save the model every 2 epochs
    if epoch % 100 == 0:
        torch.save(config.MODEL.state_dict(), f'{config.MODELS_PATH}/vae_egd_epoch_{epoch}.pth')
