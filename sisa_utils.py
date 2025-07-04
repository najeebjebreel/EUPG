import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
import torch.optim as optim
import numpy as np
import random
import copy
import tqdm
import os



# Load and preprocess data
def get_cifar10_data():
    transform_train = transforms.Compose([  
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    return trainset, testset

def get_forget_idxs(n_samples, forget_ratio):
    # Shuffle the indices to randomize the splitting
    idxs = np.arange(n_samples)
    random.shuffle(idxs)
    return idxs[:int(n_samples*forget_ratio)]

def get_shard_slices(trainset, n_shards, n_slices_per_shard):
    total_slices = n_shards*n_slices_per_shard
    n_samples= len(trainset)
    slice_size = n_samples//total_slices
    
    # Shuffle the indices to randomize the splitting
    idxs = np.arange(n_samples)
    random.shuffle(idxs)
    
    slices_per_shard = {i:[] for i in range(n_shards)}
    idxs_per_shard = {i: [] for i in range(n_shards)}

    start_idx = 0
    for j in range(total_slices):
        # Calculate indices for each slice
        start_idx = j * slice_size
        end_idx = start_idx + slice_size if j < total_slices - 1 else n_samples
        slice_idxs = idxs[start_idx:end_idx]
        slice = Subset(trainset, slice_idxs)
        shard_idx = int(j/n_slices_per_shard)
        slices_per_shard[shard_idx].append(copy.deepcopy(slice))
        idxs_per_shard[shard_idx].append(slice_idxs)  # Store the indices of this slice
    
        start_idx = end_idx  

    return idxs_per_shard, slices_per_shard
    

def get_unlearn_shard_slices(trainset, shard_slices, idxs_per_shard, forget_idxs):
    unlearn_shards_slices = {i: [] for i in range(len(shard_slices))}
    
    for shard_idx in shard_slices:
        for slice_idx, slice_idxs in enumerate(idxs_per_shard[shard_idx]):
            # Check if any of the indices in the slice should be forgotten
            if any(idx in forget_idxs for idx in slice_idxs):
                unlearn_shards_slices[shard_idx].append(slice_idx)
                
                # Find intersecting indices to forget
                intersect_idxs = np.intersect1d(slice_idxs, forget_idxs)
                
                # Remove forget indices from slice indices
                remain_slice_idxs = np.setdiff1d(slice_idxs, intersect_idxs)
                
                # Update slice indices and shard slices
                idxs_per_shard[shard_idx][slice_idx] = remain_slice_idxs
                shard_slices[shard_idx][slice_idx] = Subset(trainset, remain_slice_idxs)

    return unlearn_shards_slices, shard_slices, idxs_per_shard
                


def train_sisa_shard(model, shard_idx, shard_slices, idxs_per_shard, lr, criterion, max_epochs, batch_size, device='cuda', verbose_epoch=10):
    verbose_epoch = max(1, verbose_epoch)
    
    print(f'Training on Shard {shard_idx + 1}')
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    model.to(device)
    model.train()

    for slice_idx, slice_data in enumerate(shard_slices):
        print(f"Training on Slice {slice_idx + 1}")
        slice_loader = DataLoader(slice_data, batch_size=batch_size, shuffle=True)

        for epoch in tqdm.tqdm(range(max_epochs), desc="Epoch"):
            running_loss = 0.0

            for features, labels in slice_loader:
                features, labels = features.to(device), labels.to(device)

                # Forward pass
                outputs = model(features)
                loss = criterion(outputs, labels)

                # Backpropagation and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            running_loss /= len(slice_loader)
            scheduler.step()

            # Verbose logging every verbose_epoch epochs
            if (epoch + 1) % verbose_epoch == 0 or epoch == 0:
                print(f'Epoch [{epoch + 1}/{max_epochs}] | Train Loss: {running_loss:.4f}')

        # Save the model checkpoint after training each slice
        os.makedirs('sisa_checkpoints/cifar10', exist_ok=True)  # Ensure the directory exists
        checkpoint = {'shard_idx': shard_idx,
                      'slice_idx': slice_idx,
                      'sample_idxs': idxs_per_shard[shard_idx][slice_idx],
                      'weights': model.state_dict()}
        torch.save(checkpoint, f'sisa_checkpoints/cifar10/model_shard_{shard_idx}_slice_{slice_idx}.t7')

    return model


def unlearn_sisa_shard(model, shard_idx, unlearn_slices, shard_slices, idxs_per_shard, lr, criterion, max_epochs, batch_size, device='cuda', verbose_epoch=10):
    verbose_epoch = max(1, verbose_epoch)
    
    print(f'Unlearning Shard {shard_idx + 1}')
    print('Slices to unlearn:', np.array(unlearn_slices)+1)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    model.to(device)
    model.train()
    start_unlearn_slice = min(unlearn_slices)
    print('Start unlearning from slice', start_unlearn_slice+1)
    if start_unlearn_slice > 0:
        checkpoint = torch.load(f'sisa_checkpoints/cifar10/model_shard_{shard_idx}_slice_{start_unlearn_slice-1}.t7')
        model.load_state_dict(checkpoint['weights'])

    for slice_idx, slice_data in enumerate(shard_slices):
        if slice_idx < start_unlearn_slice:continue
        print(f"Training on Slice {slice_idx + 1}")
        slice_loader = DataLoader(slice_data, batch_size=batch_size, shuffle=True)

        for epoch in tqdm.tqdm(range(max_epochs), desc="Epoch"):
            running_loss = 0.0

            for features, labels in slice_loader:
                features, labels = features.to(device), labels.to(device)

                # Forward pass
                outputs = model(features)
                loss = criterion(outputs, labels)

                # Backpropagation and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            running_loss /= len(slice_loader)
            scheduler.step()

            # Verbose logging every verbose_epoch epochs
            if (epoch + 1) % verbose_epoch == 0 or epoch == 0:
                print(f'Epoch [{epoch + 1}/{max_epochs}] | Train Loss: {running_loss:.4f}')

        # Save the model checkpoint after training each slice
        os.makedirs('sisa_checkpoints/cifar10', exist_ok=True)  # Ensure the directory exists
        checkpoint = {'shard_idx': shard_idx,
                      'slice_idx': slice_idx,
                      'sample_idxs': idxs_per_shard[shard_idx][slice_idx],
                      'weights': model.state_dict()}
        torch.save(checkpoint, f'sisa_checkpoints/cifar10/unlearned_model_shard_{shard_idx}_slice_{slice_idx}.t7')
    
    return model



