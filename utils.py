import torch
from torch import nn
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import precision_recall_fscore_support
from pandas.api.types import CategoricalDtype

from sklearn.metrics import roc_auc_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ColumnsSelector(BaseEstimator, TransformerMixin):
    def __init__(self, type):
        self.type = type
  
    def fit(self, X, y=None):
        return self
  
    def transform(self,X):
        return X.select_dtypes(include=[self.type])
    
class CategoricalImputer(BaseEstimator, TransformerMixin):
    def __init__(self, columns = None, strategy='most_frequent'):
        self.columns = columns
        self.strategy = strategy
    
    
    def fit(self,X, y=None):
        if self.columns is None:
            self.columns = X.columns
    
        if self.strategy == 'most_frequent':
            self.fill = {column: X[column].value_counts().index[0] for column in self.columns}
        else:
            self.fill ={column: '0' for column in self.columns}
        
        return self
      
    def transform(self,X):
        X_copy = X.copy()
        for column in self.columns:
            X_copy[column] = X_copy[column].fillna(self.fill[column])
        return X_copy
  
  
class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, train_data, test_data, dropFirst=True):
        self.train_data = train_data
        self.test_data = test_data
        self.categories=dict()
        self.dropFirst=dropFirst
    
    def fit(self, X, y=None):
        join_df = pd.concat([self.train_data, self.test_data])
        join_df = join_df.select_dtypes(include=['object'])
        for column in join_df.columns:
            self.categories[column] = join_df[column].value_counts().index.tolist()
        return self
  
    def transform(self, X):
        X_copy = X.copy()
        X_copy = X_copy.select_dtypes(include=['object'])
        for column in X_copy.columns:
            X_copy[column] = X_copy[column].astype({column: CategoricalDtype(self.categories[column])})
        return pd.get_dummies(X_copy, drop_first=self.dropFirst)


def accuracy(net, loader):
    """Return accuracy on a dataset given by the data loader."""
    correct = 0
    total = 0
    for features, targets in loader:
        features, targets = features.to(DEVICE), targets.to(DEVICE)
        outputs = net(features)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return correct / total

def accuracy_with_majority_voting(nets, loader):
    """Compute accuracy using majority voting across multiple models."""
    for net in nets:
        net.eval()
        net.to(DEVICE)

    correct = 0
    total = 0
    for features, targets in loader:
        features, targets = features.to(DEVICE), targets.to(DEVICE)
        all_predictions = []
        for net in nets:
            outputs = net(features)
            _, predicted = outputs.max(1)
            all_predictions.append(predicted.unsqueeze(0))  # Add batch dimension for concatenation
        
        # Stack predictions along a new dimension to form [num_models, batch_size]
        all_predictions = torch.cat(all_predictions, dim=0)
        
        # Use mode to find the most common prediction (majority vote) for each input
        # mode returns values and indices, where values are the modes (majority votes)
        majority_votes, _ = torch.mode(all_predictions, dim=0)
        
        total += targets.size(0)
        correct += majority_votes.eq(targets).sum().item()
    
    return correct / total


def f1_score(net, loader):
    """Return F1 score on a dataset given by the data loader."""
    all_targets = []
    all_predicted = []

    for features, targets in loader:
        features, targets = features.to(DEVICE), targets.to(DEVICE)
        outputs = net(features)
        _, predicted = outputs.max(1)

        all_targets.extend(targets.cpu().numpy())
        all_predicted.extend(predicted.cpu().numpy())

    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_predicted, average='weighted')

    return f1


def auc_score(model, data_loader):
    model.eval()  # Set the model to evaluation mode

    predictions = []
    actuals = []

    with torch.no_grad():  # Turn off gradients for validation, saves memory and computations
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)  # Move data to the same device as the model
            outputs = model(inputs)

            # Move outputs to CPU for AUC computation and store them
            predictions.extend(outputs.cpu().numpy()[:, 1])
            actuals.extend(labels.cpu().numpy())

    auc_score = roc_auc_score(actuals, predictions)

    return auc_score

def auc_score_with_majority_voting(nets, loader):
    """Compute accuracy using majority voting across multiple models."""
    for net in nets:
        net.eval()
        net.to(DEVICE)

    predictions = []
    actuals = []
    for features, targets in loader:
        features, targets = features.to(DEVICE), targets.to(DEVICE)

        logits = None
        for net in nets:
            if logits is None:
                logits = net(features)
            else:
                 logits+= net(features)
        
        logits = logits/len(nets)
         # Move outputs to CPU for AUC computation and store them
        predictions.extend(logits.detach().cpu().numpy()[:, 1])
        actuals.extend(targets.detach().cpu().numpy())

    auc_score = roc_auc_score(actuals, predictions)

    return auc_score
           



def accuracy_tabnet(net, loader, idxs = None):
    """Return accuracy on a dataset given by the data loader."""
    correct = 0
    total = 0
    for features, targets in loader:
        features, targets = features.to(DEVICE), targets.to(DEVICE)
        if idxs is not None:
            outputs = net(features[:, idxs[0]], features[:, idxs[1]].long())
        else:
            outputs = net(features)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return correct / total

def compute_losses(net, loader, idxs = None):
    """Auxiliary function to compute per-sample losses"""

    criterion = nn.CrossEntropyLoss(reduction="none")
    all_losses = []

    for features, targets in loader:
        features, targets = features.to(DEVICE), targets.to(DEVICE)
        if idxs is not None:
            logits = net(features[:, idxs[0]], features[:, idxs[1]].long())
        else:
            logits = net(features)
        losses = criterion(logits, targets).detach().cpu().numpy()
        for l in losses:
            all_losses.append(l)

    return np.array(all_losses)


def compute_attack_components(net, loader, idxs=None):
    
    criterion = nn.CrossEntropyLoss(reduction="none")
    all_losses = []
    all_logits = []
    all_labels = []

    for features, targets in loader:
        features, targets = features.to(DEVICE), targets.to(DEVICE)

        if idxs is not None:
            # Assuming idxs is a tuple of indices for feature columns
            logits = net(features[:, idxs[0]], features[:, idxs[1]].long())
        else:
            logits = net(features)

        # Compute losses
        losses = criterion(logits, targets).detach().cpu().numpy()

        # Append losses to the list
        for l in losses:
            all_losses.append(l)

        # Append logits and labels to their respective lists
        all_logits.append(logits.detach().cpu().numpy())
        all_labels.append(targets.detach().cpu().numpy())

    # Concatenate logits and labels along the samples axis
    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return np.array(all_logits), np.array(all_losses), np.array(all_labels)


def compute_attack_components_sisa1(nets, loader):

    for net in nets:
        net.eval()
        net.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss(reduction="none")
    all_losses = []
    all_logits = []
    all_labels = []

    for features, targets in loader:
        features, targets = features.to(DEVICE), targets.to(DEVICE)

        logits = None
        for net in nets:
            if logits is None:
                logits = net(features)
            else:
                logits+= net(features)
        
        logits = logits/len(nets)
        # Compute losses
        losses = criterion(logits, targets).detach().cpu().numpy()

        # Append losses to the list
        for l in losses:
            all_losses.append(l)

        # Append logits and labels to their respective lists
        all_logits.append(logits.detach().cpu().numpy())
        all_labels.append(targets.detach().cpu().numpy())

    # Concatenate logits and labels along the samples axis
    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return np.array(all_logits), np.array(all_losses), np.array(all_labels)

def compute_attack_components_sisa2(nets, loaders):

    for net in nets:
        net.eval()
        net.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss(reduction="none")
    all_losses = []
    all_logits = []
    all_labels = []

    for net, loader in zip(nets, loaders):
        for features, targets in loader:
            features, targets = features.to(DEVICE), targets.to(DEVICE)
            logits = net(features)
            # Compute losses
            losses = criterion(logits, targets).detach().cpu().numpy()
            # Append losses to the list
            for l in losses:
                all_losses.append(l)

            # Append logits and labels to their respective lists
            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(targets.detach().cpu().numpy())

    # Concatenate logits and labels along the samples axis
    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return np.array(all_logits), np.array(all_losses), np.array(all_labels)

def compute_lr_attack_components(model, X, y):
    # Assuming model.predict_proba returns probabilities for each class
    probabilities = model.predict_proba(X)
    logits = np.log(probabilities)  # Convert probabilities to logits

    # Compute loss
    loss = -np.log(probabilities[np.arange(len(probabilities)), y])

    return logits, loss



def get_min_max_features(model, loader):
    seq_model = torch.nn.Sequential(*(list(model.children())[:-1]))

    # Extract features from the CIFAR10 test set
    features_list = [None for layer in seq_model]
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(loader):
            features = images.to(DEVICE)
            for idx, layer in enumerate(seq_model):
                features = layer(features)
                if batch_idx == 0:
                    features_list[idx] = features.view(features.size(0), -1)
                else:
                    features_list[idx] = torch.vstack((features_list[idx], features.view(features.size(0), -1)))
                    
        min_max_features = torch.hstack((features_list[0].min(dim = -1)[0].view(-1, 1), 
                                        features_list[0].max(dim = -1)[0].view(-1, 1)))
        for idx in range(1, len(features_list)):
            min_max_feature = torch.hstack((features_list[idx].min(dim = -1)[0].view(-1, 1), 
                                            features_list[idx].max(dim = -1)[0].view(-1, 1)))
            min_max_features = torch.hstack((min_max_features, min_max_feature))

    del features_list
    torch.cuda.empty_cache()
    return min_max_features.cpu().numpy()


def calculate_noise_scale(epsilon, delta, sensitivity=1):
    noise_scale = sensitivity * (np.sqrt(2 * np.log(1.25 / delta)) / epsilon)
    return noise_scale


def create_laplace_dp_data(data, eps):
    # Get the shape of the data matrix
    n, m = data.shape
    # Compute the sensitivity for each feature
    sensitivity = np.max(np.abs(np.diff(data, axis=0)), axis=0)
    # Create Laplace noise for each element in the matrix
    laplace_noise = np.random.laplace(scale=sensitivity/eps, size=(n, m))
    
    # Add Laplace noise to the original data
    dp_data = data + laplace_noise
    
    return dp_data

