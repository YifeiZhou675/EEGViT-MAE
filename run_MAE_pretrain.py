# modified from "https://github.com/ruiqiRichard/EEGViT/blob/master/run.py"

from models.EEGViT_MAE_pretrained import EEGViT_MAE_pretrained
from helper_functions import split
from dataset.EEGEyeNet_pretrain import EEGEyeNetDataset_pretrain
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np


# loss function
class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()

    def forward(self, x_hat, x):
        # Assuming x and x_hat have shape (batch_size, 129, 500)
        # Flatten the last two dimensions
        x_flat = x.view(x.shape[0], -1)
        x_hat_flat = x_hat.view(x_hat.shape[0], -1)
        
        # Compute the cosine similarity
        cos_sim = nn.functional.cosine_similarity(x_hat_flat, x_flat, dim=1)
        
        # Compute the loss
        loss = 1 - cos_sim
        
        # Average over the batch
        loss = torch.mean(loss)
        
        return loss


# per sample masking
def generate_masks(tensor, mask_ratio=0.7):
    """
    Generate a different mask for each sample in the batch.

    Args:
    tensor (torch.Tensor): Input tensor with shape (batch_size, num_channels, height, width)
    mask_ratio (float): Ratio of values to be masked in each sample. Should be between 0 and 1.

    Returns:
    torch.Tensor: Tensor of masks with the same shape as the input tensor.
    """
    batch_size, num_channels, height, width = tensor.shape
    num_elements = height * width

    # Calculate the number of values to be masked in each sample
    num_values_to_mask = int(num_elements * mask_ratio)

    # Initialize the mask tensor
    masks = torch.ones_like(tensor)

    # Iterate through the batch and create a mask for each sample
    for b in range(batch_size):
        for c in range(num_channels):
            # Generate random indices to mask
            indices_to_mask = torch.randperm(num_elements)[:num_values_to_mask]

            # Convert flat indices to 2D indices
            rows = indices_to_mask // width
            cols = indices_to_mask % width

            # Apply the mask
            masks[b, c, rows, cols] = 0

    return masks


def train(model, mask_ratio, criterion, optimizer, scheduler = None):
    '''
        model: model to train
        mask_ratio: ratio of elements to be masked in each sample
        criterion: metric to measure the performance of the model
        optimizer: optimizer to update weights
        scheduler: scheduling learning rate, used when finetuning pretrained models
    '''

    torch.cuda.empty_cache()
    train_indices, val_indices, test_indices = split(EEGEyeNet.trainY[:,0],0.7,0.15,0.15)  # indices for the training set
    print('create dataloader...')

    train = Subset(EEGEyeNet,indices=train_indices)
    train_loader = DataLoader(train, batch_size=batch_size)

    if torch.cuda.is_available():
        gpu_id = 0  # Change this to the desired GPU ID if you have multiple GPUs
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)  # Wrap the model with DataParallel
    print("HI")

    model = model.to(device)
    criterion = criterion.to(device)

    # Initialize lists to store losses
    train_losses = []
    print('pre-training...')
    # Train the model
    for epoch in range(n_epoch):
        model.train()
        epoch_train_loss = 0.0

        for i, (inputs, targets, index) in tqdm(enumerate(train_loader)):
            # Generate masks for each batch
            masks = generate_masks(inputs, mask_ratio=mask_ratio)

            # Move the inputs and masks to the GPU (if available)
            inputs = inputs.to(device)
            masks = masks.to(device)

            # Compute the outputs and loss for the current batch
            optimizer.zero_grad()
            outputs = model(inputs * masks)
            outputs = outputs.view(inputs.shape)
            # Compute loss on the masked entries
            loss = criterion((outputs * (1 - masks)).squeeze(), (inputs * (1 - masks)).squeeze())

            # Compute the gradients and update the parameters
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

            # Print the loss and accuracy for the current batch
            if i % 100 == 0:
                print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item()}")

        epoch_train_loss /= len(train_loader)
        train_losses.append(epoch_train_loss)

        if scheduler is not None:
            scheduler.step()


if __name__ == "__main__":
    model = EEGViT_MAE_pretrained()
    EEGEyeNet = EEGEyeNetDataset_pretrain('./dataset/Position_task_with_dots_synchronised_min_88.npz')
    batch_size = 64
    n_epoch = 30
    learning_rate = 1e-4

    criterion = CosineSimilarityLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    mask_ratio = 0.5
    train(model, mask_ratio=mask_ratio, criterion=criterion, optimizer=optimizer, scheduler=scheduler)
    torch.save(model.state_dict(), f'./models_weights/pretrained_mask_ratio_{mask_ratio}.pth')
