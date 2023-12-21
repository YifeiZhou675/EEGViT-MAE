from models.EEGViTPre_finetune import EEGViTPre_finetune
from helper_functions import split
from dataset.EEGEyeNet import EEGEyeNetDataset
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
import argparse


def train(model: EEGViTPre_finetune, EEGEyeNet, criterion, optimizer, scheduler=None,
          batch_size=64, n_epoch=15):
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

    train = Subset(EEGEyeNet, indices=train_indices)
    val = Subset(EEGEyeNet, indices=val_indices)
    test = Subset(EEGEyeNet, indices=test_indices)

    train_loader = DataLoader(train, batch_size=batch_size)
    val_loader = DataLoader(val, batch_size=batch_size)
    test_loader = DataLoader(test, batch_size=batch_size)

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
    criterion = criterion.to(device)

    # Initialize lists to store losses
    train_losses = []
    val_losses = []
    test_losses = []
    print('fine-tuning...')
    # Train the model
    for epoch in range(n_epoch):
        model.train()
        epoch_train_loss = 0.0

        for i, (inputs, targets, index) in tqdm(enumerate(train_loader)):
            # Move the inputs and targets to the GPU (if available)
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Compute the outputs and loss for the current batch
            optimizer.zero_grad()
            outputs = model(eeg=inputs)
            loss = criterion(outputs.squeeze(), targets.squeeze())

            # Compute the gradients and update the parameters
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

            # Print the loss and accuracy for the current batch
            if i % 100 == 0:
                print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item()}")

        epoch_train_loss /= len(train_loader)
        train_losses.append(epoch_train_loss)

        # Evaluate the model on the validation set
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for inputs, targets, index in val_loader:
                # Move the inputs and targets to the GPU (if available)
                inputs = inputs.to(device)
                targets = targets.to(device)

                # Compute the outputs and loss for the current batch
                outputs = model(inputs)
                # print(outputs)
                loss = criterion(outputs.squeeze(), targets.squeeze())
                val_loss += loss.item()

            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            print(f"Epoch {epoch}, Val Loss: {val_loss}")

        with torch.no_grad():
            val_loss = 0.0
            for inputs, targets, index in test_loader:
                # Move the inputs and targets to the GPU (if available)
                inputs = inputs.to(device)
                targets = targets.to(device)

                # Compute the outputs and loss for the current batch
                outputs = model(inputs)

                loss = criterion(outputs.squeeze(), targets.squeeze())
                val_loss += loss.item()

            val_loss /= len(test_loader)
            test_losses.append(val_loss)

            print(f"Epoch {epoch}, test Loss: {val_loss}")

        if scheduler is not None:
            scheduler.step()


def get_args_parser():
    parser = argparse.ArgumentParser('EEGViT-MAE fine-tuning', add_help=False)

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--n_epoch', default=15, type=int)
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--data_path', default='./dataset/Position_task_with_dots_synchronised_min.npz', type=str)
    parser.add_argument('--model_path', required=True, type=str, help='path to save fine-tuned models')
    parser.add_argument('--weight_path', required=True, type=str, help='path to load pre-trained models')
    parser.add_argument('--seed', default=0, type=int, help='random seed')

    return parser


def main(args):
    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = EEGViTPre_finetune()
    model.load_pretrained(args.weight_path)
    EEGEyeNet = EEGEyeNetDataset(args.data_path)
    batch_size = args.batch_size
    n_epoch = args.n_epoch
    learning_rate = args.learning_rate
    criterion = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)

    train(model=model, EEGEyeNet=EEGEyeNet, criterion=criterion, optimizer=optimizer,
          scheduler=scheduler, batch_size=batch_size, n_epoch=n_epoch)
    torch.save(model.state_dict(), args.model_path)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
