import wandb
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from dataloader import EGDDataset
from efficientnet import initialize_model
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize wandb
wandb.init(project='gastroscopy_classifier_training_VAE_data', entity='neildlf')

# Initialize dataset
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.Resize((560, 640)),
    transforms.ToTensor(),
])

root_dir = "/home/ndelafuente/CVC/EGD_Barcelona/gastroscopy_attention_classifier/TRAIN&MODEL_4_GENERATEDDATA/data"

#set cuda device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create dataset for training
train_dataset = EGDDataset(
    root_dir=root_dir,
    transform=transform,
    use_test_data=False 
)

# Create dataset for testing
test_dataset = EGDDataset(
    root_dir=root_dir,
    transform=transform,
    use_test_data=True,
    only_test_data=True
)

#Compute class weights for the loss function (to deal with unbalanced dataset)
class_counts = np.bincount(train_dataset.labels)
class_weights = 1. / torch.tensor(class_counts, dtype=torch.float32)

# Hyperparameters
num_epochs = 230
learning_rate = 5e-4
num_classes = 3
batch_size = 12

# Initialize sum of state_dicts to zero, for calculating mean later
sum_state_dict = None

# Initialize dictionary to store metrics for each fold
metrics_dict = {}

best_val_loss = float('inf')    

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

model = initialize_model(num_classes=num_classes).to(device)
'''
    for name, param in model.named_parameters():
        if 'weight' in name:
            if len(param.size()) > 1:
                nn.init.xavier_uniform_(param)
'''       
'''
If i wanted to initalize the model with the pretrained way, uncomment next 2 lines
'''
    #model_weights_path = "/home/ndelafuente/CVC/EGD_Barcelona/gastroscopy_attention_classifier/kfold_weights/model_fold_3.pth"
    #model.load_state_dict(torch.load(model_weights_path))
    
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

train_losses = []
val_losses = []
val_accuracies = []

confusion_matrices = []

conf_matrix = np.zeros((num_classes, num_classes))

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    train_losses.append(running_loss / len(train_loader))
    
    model.eval()
    val_loss = 0.0
    correct, total = 0, 0
    
    conf_mat = np.zeros((num_classes, num_classes))
    torch.cuda.empty_cache()
    
    with torch.no_grad():
        print(device)
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            conf_mat += confusion_matrix(labels.cpu().numpy(), predicted.cpu().numpy())
            torch.cuda.empty_cache()
    
    val_loss /= len(test_loader)
    val_losses.append(val_loss)
    val_accuracies.append(100*correct / total)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), f"/home/ndelafuente/CVC/EGD_Barcelona/gastroscopy_attention_classifier/TRAIN&MODEL_4_GENERATEDDATA/best_model.pth")
        
    print(f"Epoch {epoch+1}/{num_epochs} | Train loss: {train_losses[-1]:.4f} | Val loss: {val_losses[-1]:.4f} | Val acc: {val_accuracies[-1]:.2f}%")
    print(f"correcly classified = {correct}, incorrectly classified = {total-correct}")
    
    if epoch > 100:
        confusion_matrices.append(conf_mat)
        
    plt.figure(figsize=(8,6))
    sns.heatmap(conf_mat, annot=True, fmt='g', cmap='cool')
    plt.title('Confusion matrix')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    wandb.log({f"confusion_matrix_{epoch}": wandb.Image(plt)})
    
    class_recalls = []
    
    class_metrics = {}
    
    for i in range(num_classes):
        tp = conf_mat[i, i]
        fp = np.sum(conf_mat[:, i]) - tp
        fn = np.sum(conf_mat[i, :]) - tp
        tn = np.sum(conf_mat) - tp - fp - fn
        
        accuracy = (tp + np.sum(np.diag(conf_mat)) - tp) / np.sum(conf_mat)
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn)    
        f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        
        class_metrics[f"class_{i}_accuracy"] = accuracy
        class_metrics[f"class_{i}_precision"] = precision
        class_metrics[f"class_{i}_recall"] = recall
        class_metrics[f"class_{i}_f1_score"] = f1_score
        
        class_recalls.append(recall) #append the recall for this class
        
        print(f"class {i} accuracy = {accuracy*100:.2f}%,\
              precision = {precision*100:.2f}%,\
              recall = {recall*100:.2f}%,\
              f1_score = {f1_score*100:.2f}%\n")
        
    global_accuracy = np.trace(conf_mat) / np.sum(conf_mat)
    global_recall = np.mean(class_recalls)
    global_f1_score = 2 * global_recall * global_accuracy / (global_recall + global_accuracy)
    
    class_metrics["global_accuracy"] = global_accuracy
    class_metrics["global_recall"] = global_recall
    class_metrics["global_f1_score"] = global_f1_score
    
    print(f"global accuracy = {global_accuracy*100:.2f}%,\
          global recall = {global_recall*100:.2f}%,\
          global f1_score = {global_f1_score*100:.2f}%\n")
    
    wandb.log({
        "train_loss": train_losses[-1],
        "val_loss": val_losses[-1],
        "val_acc": val_accuracies[-1],
        "global_accuracy": global_accuracy,
        "global_recall": global_recall,
        "global_f1_score": global_f1_score,
        "class_metrics": class_metrics
    })
    
    metrics_dict[f"epoch_{epoch}"] = {
        "train_loss": train_losses[-1],
        "val_loss": val_losses[-1],
        "val_acc": val_accuracies[-1],
        "global_accuracy": global_accuracy,
        "global_recall": global_recall,
        "global_f1_score": global_f1_score,
        "class_metrics": class_metrics
    }

if sum_state_dict is not None:
    for key in sum_state_dict:
        sum_state_dict[key] += model.state_dict()[key]
else:
    sum_state_dict = model.state_dict()
    
# Calculate the average confusion matrix
avg_conf_mat = np.mean(confusion_matrices, axis=0)

#Plot the average confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(avg_conf_mat, annot=True, fmt='g', cmap='cool')
plt.title('Average Confusion matrix')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.savefig('avg_confusion_matrix.png')
plt.close()
wandb.log({"avg_confusion_matrix": wandb.Image(plt)})

wandb.log(metrics_dict)

print(metrics_dict)

wandb.finish()
