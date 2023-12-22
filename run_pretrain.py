from models.EEGViTPre_pretrain import EEGViTPre_pretrain
from helper_functions import split
from dataset.EEGEyeNet import EEGEyeNetDataset
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
import argparse


# per sample masking
def generate_masks(eeg, mask_ratio=0.5):
    """
    Generate a different mask for each sample in the batch.

    Args:
    eeg (torch.Tensor): Input tensor with shape (batch_size, num_channels, height, width)
    mask_ratio (float): Ratio of values to be masked in each sample. Should be between 0 and 1.

    Returns:
    torch.Tensor: Tensor of masks with the same shape as the input tensor.
    """
    batch_size, num_channels, height, width = eeg.shape
    num_elements = height * width

    # Calculate the number of values to be masked in each sample
    num_values_to_mask = int(num_elements * mask_ratio)

    # Initialize the mask tensor
    masks = torch.ones_like(eeg)

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


def train(model: EEGViTPre_pretrain, EEGEyeNet, mask_ratio, optimizer, scheduler=None,
          batch_size=64, n_epoch=30):
    """
        model: model to train
        mask_ratio: ratio of elements to be masked in each sample
        criterion: metric to measure the performance of the model
        optimizer: optimizer to update weights
        scheduler: scheduling learning rate, used when fine-tuning pretrained models
    """

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
        model = torch.nn.DataParallel(model)  # Wrap the model with DataParallel
    print("HI")

    model = model.to(device)

    # Initialize lists to store losses
    train_losses = []
    print('pre-training...')
    # Train the model
    for epoch in range(n_epoch):
        model.train()
        epoch_train_loss = 0.0

        for i, (inputs, targets, index) in tqdm(enumerate(train_loader)):
            # Generate masks for each batch
            masks = generate_masks(eeg=inputs, mask_ratio=mask_ratio)

            # Move the inputs and masks to the GPU (if available)
            inputs = inputs.to(device)
            masks = masks.to(device)

            # Compute the outputs and loss for the current batch
            optimizer.zero_grad()
            loss = model(eeg=inputs, mask=masks)

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


def get_args_parser():
    parser = argparse.ArgumentParser('EEGViT-MAE pre-training', add_help=False)

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--n_epoch', default=30, type=int)
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--mask_ratio', default=0.5, type=float)
    parser.add_argument('--data_path', default='./dataset/Position_task_with_dots_synchronised_min.npz', type=str)
    parser.add_argument('--model_path', required=True, type=str, help='path to save pre-trained models')
    parser.add_argument('--seed', default=0, type=int, help='random seed')

    return parser


def main(args):
    model = EEGViTPre_pretrain()
    EEGEyeNet = EEGEyeNetDataset(args.data_path)
    batch_size = args.batch_size
    n_epoch = args.n_epoch
    learning_rate = args.learning_rate

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    mask_ratio = args.mask_ratio
    train(model=model, EEGEyeNet=EEGEyeNet, mask_ratio=mask_ratio, optimizer=optimizer,
          scheduler=scheduler, batch_size=batch_size, n_epoch=n_epoch)
    torch.save(model.state_dict(), args.model_path)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
