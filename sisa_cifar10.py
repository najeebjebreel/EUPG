from sisa_utils import *
from utils import *
from datasets import *
from mdav import *
from train import *
from models import *
from attacks import *
from dp_data.load_dp_cifar10_dataset import *

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset

import random
import time
import copy


import csv

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import warnings
from sklearn.exceptions import ConvergenceWarning, FitFailedWarning

# Filter out ConvergenceWarning and FitFailedWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FitFailedWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def seed_everything(seed=7):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(seed=7)

trainset, testset = get_cifar10_data()

forget_ratio = 0.01
forget_idxs = get_forget_idxs(len(trainset), forget_ratio)
forget_set = Subset(trainset, forget_idxs)
print('Number of samples to forget:', len(forget_set))
# SISA shards and slices
n_shards = 5
n_slices_per_shard = 10  # Added: Number of slices per shard
idxs_per_shard, shard_slices = get_shard_slices(trainset, n_shards, n_slices_per_shard)

# Check if everything is fine
# print(len(slice_samples_idxs))
# print(len(shard_slices))
# print(len(shard_slices[0]))
# print(len(shard_slices[0][0]))

# Model setup
num_classes = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
initial_model = densenet(num_classes=num_classes, depth=100, growthRate=12, compressionRate=2, dropRate=0)
criterion = nn.CrossEntropyLoss()
lr = 0.1
n_repeat = 1
max_epochs = 1
batch_size = 64

train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
forget_loader = DataLoader(forget_set, batch_size=batch_size, shuffle=False)


print('Training original models on shards and slices...')
print(f'#Shards {n_shards}, #Slices {n_slices_per_shard}')
# Define and train M on D with SISA
train_accs = []
test_accs = []
mia_aucs = []
mia_advs = []
runtimes = []
for r in range(n_repeat):
    torch.cuda.empty_cache()
    models = []
    t0 = time.time()
    for shard_idx in range(n_shards):
        shard_model = copy.deepcopy(initial_model)
        shard_model = train_sisa_shard(shard_model, shard_idx, shard_slices[shard_idx], idxs_per_shard[shard_idx], 
                                       lr, criterion, max_epochs, batch_size, 
                                       device=device, verbose_epoch = int(max_epochs/10))
        models.append(shard_model)
        print('---------------------------------------------')
    t1 = time.time()
    rt = t1-t0
    runtimes.append(rt)

    # Evaluate the model accuracy, and MIA
    # Accuracy
    train_acc = accuracy_with_majority_voting(models, train_loader)
    test_acc = accuracy_with_majority_voting(models, test_loader)
    train_accs.append(100.0*train_acc)
    test_accs.append(100.0*test_acc)
    #MIA
    idxs = np.arange(len(testset))
    random.shuffle(idxs)
    m = len(forget_set)
    rand_idxs = idxs[:m]
    logits_test, loss_test, test_labels = compute_attack_components_sisa1(models, test_loader)
    logits_forget, loss_forget, forget_labels = compute_attack_components_sisa1(models, forget_loader)
    attack_result = tf_attack(logits_forget, logits_test[rand_idxs], loss_forget, loss_test[rand_idxs], 
                          forget_labels, test_labels[rand_idxs])
    auc = attack_result.get_result_with_max_auc().get_auc()
    adv = attack_result.get_result_with_max_attacker_advantage().get_attacker_advantage()
    mia_aucs.append(100.0*auc)
    mia_advs.append(100.0*adv)

mean_runtime = np.mean(runtimes)
std_runtime = np.std(runtimes)
mean_train_acc = np.mean(train_accs)
std_train_acc = np.std(train_accs)
mean_test_acc = np.mean(test_accs)
std_test_acc = np.std(test_accs)
mean_mia_auc = np.mean(mia_aucs)
std_mia_auc = np.std(mia_aucs)
mean_mia_adv = np.mean(mia_advs)
std_mia_adv = np.std(mia_advs)

# Print the results
print('Training M on D time:{:0.2f}(±{:0.2f}) seconds'.format(mean_runtime, std_runtime))
print('Train accuracy:{:0.2f}(±{:0.2f})%'.format(mean_train_acc, std_train_acc))
print('Test accuracy:{:0.2f}(±{:0.2f})%'.format(mean_test_acc, std_test_acc))
print('MIA AUC:{:0.2f}(±{:0.2f})%'.format(mean_mia_auc, std_mia_auc))
print('MIA Advantage:{:0.2f}(±{:0.2f})%'.format(mean_mia_adv, std_mia_adv))

# Save to CSV
csv_file_path = 'results/SISA/cifar10/densenet_shards={}_slices={}_fr={}_base.csv'.format(n_shards, n_slices_per_shard, forget_ratio)

with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Metric', 'Mean', 'Standard Deviation'])
    writer.writerow(['Training Time', mean_runtime, std_runtime])
    writer.writerow(['Train accuracy', mean_train_acc, std_train_acc])
    writer.writerow(['Test accuracy', mean_test_acc, std_test_acc])
    writer.writerow(['MIA AUC', mean_mia_auc, std_mia_auc])
    writer.writerow(['MIA Advantage', mean_mia_adv, std_mia_adv])

del models

print('----------------------------------------------')
print('Number of samples to forget:', len(forget_set))
print('Unlearn from identified shards and slices...')
# Unlearn Df with SISA
unlearn_shards_slices, shard_slices, idxs_per_shard = get_unlearn_shard_slices(trainset, shard_slices, idxs_per_shard, forget_idxs)

# Retrain on Dr with SISA
train_accs = []
test_accs = []
mia_aucs = []
mia_advs = []
runtimes = []
for r in range(n_repeat):
    torch.cuda.empty_cache()
    models = []
    t0 = time.time()
    for shard_idx in range(n_shards):
        shard_model = copy.deepcopy(initial_model)
        if len(unlearn_shards_slices[shard_idx]) < 1:
            print(f'Shard {shard_idx+1} has no sample to unlearn')
            checkpoint = torch.load(f'sisa_checkpoints/cifar10/model_shard_{shard_idx}_slice_{n_slices_per_shard-1}.t7')
            shard_model.load_state_dict(checkpoint['weights'])
        else:
            shard_model = unlearn_sisa_shard(shard_model, shard_idx, unlearn_shards_slices[shard_idx], shard_slices[shard_idx], idxs_per_shard[shard_idx], 
                                       lr, criterion, max_epochs, batch_size, 
                                       device=device, verbose_epoch = int(max_epochs/10))
        models.append(shard_model)
        print('---------------------------------------------')
    t1 = time.time()
    rt = t1-t0
    runtimes.append(rt)

    # Evaluate the model accuracy, and MIA
    # Accuracy
    train_acc = accuracy_with_majority_voting(models, train_loader)
    test_acc = accuracy_with_majority_voting(models, test_loader)
    train_accs.append(100.0*train_acc)
    test_accs.append(100.0*test_acc)
    #MIA
    idxs = np.arange(len(testset))
    random.shuffle(idxs)
    m = len(forget_set)
    rand_idxs = idxs[:m]
    logits_test, loss_test, test_labels = compute_attack_components_sisa1(models, test_loader)
    logits_forget, loss_forget, forget_labels = compute_attack_components_sisa1(models, forget_loader)
    attack_result = tf_attack(logits_forget, logits_test[rand_idxs], loss_forget, loss_test[rand_idxs], 
                          forget_labels, test_labels[rand_idxs])
    auc = attack_result.get_result_with_max_auc().get_auc()
    adv = attack_result.get_result_with_max_attacker_advantage().get_attacker_advantage()
    mia_aucs.append(100.0*auc)
    mia_advs.append(100.0*adv)

mean_runtime = np.mean(runtimes)
std_runtime = np.std(runtimes)
mean_train_acc = np.mean(train_accs)
std_train_acc = np.std(train_accs)
mean_test_acc = np.mean(test_accs)
std_test_acc = np.std(test_accs)
mean_mia_auc = np.mean(mia_aucs)
std_mia_auc = np.std(mia_aucs)
mean_mia_adv = np.mean(mia_advs)
std_mia_adv = np.std(mia_advs)

# Print the results
print('Training M on D time:{:0.2f}(±{:0.2f}) seconds'.format(mean_runtime, std_runtime))
print('Train accuracy:{:0.2f}(±{:0.2f})%'.format(mean_train_acc, std_train_acc))
print('Test accuracy:{:0.2f}(±{:0.2f})%'.format(mean_test_acc, std_test_acc))
print('MIA AUC:{:0.2f}(±{:0.2f})%'.format(mean_mia_auc, std_mia_auc))
print('MIA Advantage:{:0.2f}(±{:0.2f})%'.format(mean_mia_adv, std_mia_adv))

# Save to CSV
csv_file_path = 'results/SISA/cifar10/densenet_shards={}_slices={}_fr={}_base.csv'.format(n_shards, n_slices_per_shard, forget_ratio)

with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Metric', 'Mean', 'Standard Deviation'])
    writer.writerow(['Training Time', mean_runtime, std_runtime])
    writer.writerow(['Train accuracy', mean_train_acc, std_train_acc])
    writer.writerow(['Test accuracy', mean_test_acc, std_test_acc])
    writer.writerow(['MIA AUC', mean_mia_auc, std_mia_auc])
    writer.writerow(['MIA Advantage', mean_mia_adv, std_mia_adv])

del models
