import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader

import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits

from sklearn.preprocessing import LabelEncoder

import pickle


np.random.seed(0)

def generate_synthetic_data(num_samples = 1000, num_features = 2, num_classes = 2):
    # Generate synthetic features and labels
    X, y = make_classification(n_samples=num_samples, n_features=num_features, n_redundant=0, n_repeated=0, 
                            n_informative=num_features, n_classes=num_classes, n_clusters_per_class=num_classes, 
                            random_state=7) #for reproducibility 


    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=7
    )

    # Create TensorDatasets
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.int64))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.int64))
    

    return X_train, y_train, X_test, y_test, train_dataset, test_dataset

def get_digit_dataset():
    
    # Step 1: Get the digit one dataset from scikit-learn
    digits = load_digits()
    X, y = digits.data, digits.target

    # Split the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create TensorDatasets
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.int64))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.int64))
    

    return X_train, y_train, X_test, y_test, train_dataset, test_dataset



def generate_dataset_from_mdav_results(clusters, labels):
    X = None
    y = None
    for i in range(len(labels)):
        xc = clusters[i]
        yc = labels[i]
        if X is None:
            X = xc
            y = yc
        else:
            X = np.vstack((X, xc))
            y = np.hstack((y, yc))
    
    return X, y

def get_anonymized_adult(k, batch_size, with_class = ''):
    # Specify the file names for loading
    file_X_train_k = "data/adult/X_train_{}_k={}.csv".format(with_class, k)
    file_y_train_k = "data/adult/y_train_{}_k={}.pkl".format(with_class, k)
    # Load the variables from the binary files
    X_train_k = pd.read_csv(file_X_train_k)
    X_train_k['age'] = X_train_k['age'].round(0).astype(int)
    X_train_k['education-num'] = X_train_k['education-num'].round(0).astype(int)
    X_train_k['hours-per-week'] = X_train_k['hours-per-week'].round(0).astype(int)
        
    with open(file_y_train_k, 'rb') as f:
        y_train_k = pickle.load(f)
    
    if with_class == 'nc':
        y_train_k = (y_train_k == '>50K') + 0
        for col in X_train_k.columns:
            l_enc = LabelEncoder()
            X_train_k[col] = l_enc.fit_transform(X_train_k[col].values)
    else:
        for col in X_train_k.columns:
            l_enc = LabelEncoder()
            X_train_k[col] = l_enc.fit_transform(X_train_k[col].values)

   
    # Create TensorDatasets
    train_dataset_k = TensorDataset(torch.tensor(X_train_k.values, dtype=torch.float32), torch.tensor(y_train_k, dtype=torch.int64))
    train_loader_k = DataLoader(train_dataset_k, batch_size=batch_size, shuffle=True)

    return train_loader_k


# Define DP-dataset class
class DPDataset(Dataset):
    def __init__(self, data, noise_scale):
        self.data = data
        self.noise_scale = noise_scale

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        noise = torch.FloatTensor(self.data[idx][0].shape).normal_(0, self.noise_scale)
        return self.data[idx][0] + noise, self.data[idx][1]
    