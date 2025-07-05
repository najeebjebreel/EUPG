#!/usr/bin/env python3
"""
Unified benchmark script for machine learning with privacy guarantees.
Supports multiple datasets, models, and privacy techniques (k-anonymity, differential privacy, SISA).
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import datasets, transforms
import random
import time
import copy
import csv
from collections import Counter
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn import metrics
import xgboost as xgb
from xgboost import XGBClassifier
import warnings
import os
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion

# Import custom modules (assumed to exist)
from utils import *
from datasets import *
from mdav import *
from train import *
from models import *
from attacks import *
from dp_data.load_dp_cifar10_dataset import *

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def seed_everything(seed=7):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class DatasetLoader:
    """Unified dataset loading class"""
    
    @staticmethod
    def load_adult():
        columns = ["age", "workClass", "fnlwgt", "education", "education-num",
                   "marital-status", "occupation", "relationship", "race", "sex", 
                   "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
        
        train_data = pd.read_csv('data/adult/adult.data', names=columns, sep=r' *, *', 
                                engine='python', na_values='?')
        test_data = pd.read_csv('data/adult/adult.test', names=columns, sep=r' *, *', 
                               skiprows=1, engine='python', na_values='?')
        
        # Preprocessing pipeline
        num_pipeline = Pipeline(steps=[
            ("num_attr_selector", ColumnsSelector(type='int')),
            ("scaler", StandardScaler())
        ])
        cat_pipeline = Pipeline(steps=[
            ("cat_attr_selector", ColumnsSelector(type='object')),
            ("cat_imputer", CategoricalImputer(columns=['workClass','occupation', 'native-country'])),
            ("encoder", CategoricalEncoder(train_data, test_data, dropFirst=True))
        ])
        full_pipeline = FeatureUnion([("num_pipe", num_pipeline), ("cat_pipeline", cat_pipeline)])
        
        # Clean and process data
        for df in [train_data, test_data]:
            df.drop(['fnlwgt', 'education'], axis=1, inplace=True)
            df.dropna(inplace=True)
        
        train_copy = train_data.copy()
        train_copy["income"] = train_copy["income"].apply(lambda x: 0 if x=='<=50K' else 1)
        X_train = train_copy.drop('income', axis=1)
        y_train = train_copy['income'].values
        X_train = full_pipeline.fit_transform(X_train)
        
        test_copy = test_data.copy()
        test_copy["income"] = test_copy["income"].apply(lambda x: 0 if x=='<=50K.' else 1)
        X_test = test_copy.drop('income', axis=1)
        y_test = test_copy['income'].values
        X_test = full_pipeline.transform(X_test)
        
        return X_train, y_train, X_test, y_test, full_pipeline
    
    @staticmethod
    def load_credit():
        df_train = pd.read_csv('data/GiveMeSomeCredit/cs-training.csv')
        df_train.drop(columns=['Unnamed: 0'], inplace=True)
        df_train.dropna(inplace=True)
        y = df_train['SeriousDlqin2yrs'].values
        df_train.drop(['SeriousDlqin2yrs'], axis=1, inplace=True)
        X = df_train.values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        SC = StandardScaler()
        X_train = SC.fit_transform(X_train)
        X_test = SC.transform(X_test)
        
        return X_train, y_train, X_test, y_test, SC
    
    @staticmethod
    def load_heart():
        df = pd.read_csv('data/heart/cardio_train.csv', sep=';')
        df.drop(columns=['id'], inplace=True)
        df.dropna(inplace=True)
        
        mask = np.random.rand(len(df)) < 0.8
        trainset = df[mask].reset_index(drop=True)
        testset = df[~mask].reset_index(drop=True)
        
        X_train = trainset.iloc[:,:-1].values
        y_train = trainset.iloc[:,-1].values
        X_test = testset.iloc[:,:-1].values
        y_test = testset.iloc[:,-1].values
        
        SC = StandardScaler()
        X_train = SC.fit_transform(X_train)
        X_test = SC.transform(X_test)
        
        return X_train, y_train, X_test, y_test, SC
    
    @staticmethod
    def load_cifar10():
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

class ModelFactory:
    """Factory for creating different model types"""
    
    @staticmethod
    def create_model(model_type, dataset, num_features=None, num_classes=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if model_type == 'mlp':
            hidden_size = 128 if dataset != 'credit' else 256
            return MLPModel(num_features, hidden_size, num_classes)
        elif model_type == 'xgboost':
            if dataset == 'adult':
                return XGBClassifier(num_classes=num_classes, reg_lambda=5, 
                                   learning_rate=0.5, max_depth=10, n_estimators=300, device=device)
            elif dataset == 'credit':
                return XGBClassifier(num_classes=num_classes, reg_lambda=5, 
                                   learning_rate=0.5, max_depth=9, n_estimators=200, device=device)
            elif dataset == 'heart':
                return XGBClassifier(num_classes=num_classes, reg_lambda=5, 
                                   learning_rate=0.5, max_depth=7, n_estimators=200, device=device)
        elif model_type == 'densenet':
            return densenet(num_classes=num_classes, depth=100, growthRate=12, compressionRate=2, dropRate=0)
        
        raise ValueError(f"Unknown model type: {model_type}")

class PrivacyBenchmark:
    """Main benchmarking class"""
    
    def __init__(self, dataset, model_type, forget_ratios, n_repeat=3, max_epochs=100):
        seed_everything()
        self.dataset = dataset
        self.model_type = model_type
        self.forget_ratios = forget_ratios
        self.n_repeat = n_repeat
        self.max_epochs = max_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = OneHotEncoder(sparse_output=False, categories="auto")
        
        # Load dataset
        self._load_data()
        
        # Setup model
        self._setup_model()
        
        # Create results directory
        os.makedirs(f'results/{dataset}', exist_ok=True)
    
    def _load_data(self):
        """Load and prepare dataset"""
        if self.dataset == 'adult':
            self.X_train, self.y_train, self.X_test, self.y_test, self.preprocessor = DatasetLoader.load_adult()
            self.batch_size = 512
            self.is_tabular = True
        elif self.dataset == 'credit':
            self.X_train, self.y_train, self.X_test, self.y_test, self.preprocessor = DatasetLoader.load_credit()
            self.batch_size = 256
            self.is_tabular = True
        elif self.dataset == 'heart':
            self.X_train, self.y_train, self.X_test, self.y_test, self.preprocessor = DatasetLoader.load_heart()
            self.batch_size = 512
            self.is_tabular = True
        elif self.dataset == 'cifar10':
            self.trainset, self.testset = DatasetLoader.load_cifar10()
            self.y_train = self.trainset.targets
            self.batch_size = 64
            self.is_tabular = False
        
        if self.is_tabular:
            self.num_features = self.X_train.shape[1]
            self.num_classes = len(set(self.y_train))
            self._create_data_loaders()
        else:
            self.num_classes = 10
            self.num_features = None
    
    def _create_data_loaders(self):
        """Create PyTorch data loaders for tabular data"""
        self.train_dataset = TensorDataset(
            torch.tensor(self.X_train, dtype=torch.float32), 
            torch.tensor(self.y_train, dtype=torch.int64)
        )
        self.test_dataset = TensorDataset(
            torch.tensor(self.X_test, dtype=torch.float32), 
            torch.tensor(self.y_test, dtype=torch.int64)
        )
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
    
    def _setup_model(self):
        """Setup initial model"""
        self.initial_model = ModelFactory.create_model(
            self.model_type, self.dataset, self.num_features, self.num_classes
        )
        
        # Set learning rate and criterion based on model type
        if self.model_type == 'mlp':
            self.lr = 1e-2 if self.dataset == 'heart' else 1e-3 if self.dataset == 'credit' else 1e-2
            self.criterion = nn.CrossEntropyLoss()
        elif self.model_type == 'densenet':
            self.lr = 0.1
            self.criterion = nn.CrossEntropyLoss()
        # XGBoost doesn't use these parameters
    
    def _split_forget_retain(self, forget_ratio):
        """Split data into forget and retain sets"""
        if self.is_tabular:
            m = int(len(self.y_train) * forget_ratio)
            idxs = np.arange(len(self.y_train))
            random.shuffle(idxs)
            
            retain_idxs = idxs[m:]
            forget_idxs = idxs[:m]
            
            X_retain = self.X_train[retain_idxs]
            y_retain = self.y_train[retain_idxs]
            X_forget = self.X_train[forget_idxs]
            y_forget = self.y_train[forget_idxs]
            
            retain_dataset = TensorDataset(
                torch.tensor(X_retain, dtype=torch.float32), 
                torch.tensor(y_retain, dtype=torch.int64)
            )
            forget_dataset = TensorDataset(
                torch.tensor(X_forget, dtype=torch.float32), 
                torch.tensor(y_forget, dtype=torch.int64)
            )
            
            retain_loader = DataLoader(retain_dataset, batch_size=self.batch_size, shuffle=True)
            forget_loader = DataLoader(forget_dataset, batch_size=self.batch_size, shuffle=False)
            
            return X_retain, y_retain, X_forget, y_forget, retain_loader, forget_loader, m
        else:
            # CIFAR-10 case
            m = int(len(self.trainset) * forget_ratio)
            idxs = np.arange(len(self.trainset))
            random.shuffle(idxs)
            
            retain_idxs = idxs[m:]
            forget_idxs = idxs[:m]
            
            retain_set = Subset(self.trainset, retain_idxs)
            forget_set = Subset(self.trainset, forget_idxs)
            
            train_loader = DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)
            test_loader = DataLoader(self.testset, batch_size=self.batch_size, shuffle=False)
            retain_loader = DataLoader(retain_set, batch_size=self.batch_size, shuffle=True)
            forget_loader = DataLoader(forget_set, batch_size=self.batch_size, shuffle=False)
            
            return None, None, None, None, retain_loader, forget_loader, m
    
    def _evaluate_model(self, model, is_pytorch=True):
        """Evaluate model performance"""
        if is_pytorch:
            model.eval()
            if self.dataset == 'credit':
                train_acc = auc_score(model, self.train_loader)
                test_acc = auc_score(model, self.test_loader)
            else:
                train_acc = accuracy(model, self.train_loader if self.is_tabular else DataLoader(self.trainset, batch_size=self.batch_size, shuffle=False))
                test_acc = accuracy(model, self.test_loader if self.is_tabular else DataLoader(self.testset, batch_size=self.batch_size, shuffle=False))
        else:
            # XGBoost
            if self.dataset == 'credit':
                train_acc = roc_auc_score(self.y_train, model.predict_proba(self.X_train)[:, 1])
                test_acc = roc_auc_score(self.y_test, model.predict_proba(self.X_test)[:, 1])
            else:
                train_acc = accuracy_score(self.y_train, model.predict(self.X_train))
                test_acc = accuracy_score(self.y_test, model.predict(self.X_test))
        
        return train_acc, test_acc
    
    def _compute_mia(self, model, forget_loader, m, is_pytorch=True):
        """Compute membership inference attack metrics"""
        if is_pytorch:
            test_loader = self.test_loader if self.is_tabular else DataLoader(self.testset, batch_size=self.batch_size, shuffle=False)
            idxs = np.arange(len(self.test_dataset if self.is_tabular else self.testset))
            random.shuffle(idxs)
            rand_idxs = idxs[:m]
            
            logits_test, loss_test, test_labels = compute_attack_components(model, test_loader)
            logits_forget, loss_forget, forget_labels = compute_attack_components(model, forget_loader)
            attack_result = tf_attack(logits_forget, logits_test[rand_idxs], loss_forget, loss_test[rand_idxs], 
                                    forget_labels, test_labels[rand_idxs])
        else:
            # XGBoost MIA
            test_preds = model.predict_proba(self.X_test)
            forget_preds = model.predict_proba(self.X_forget)
            
            y_test_one_hot = self.encoder.fit_transform(self.y_test.reshape(-1, 1))
            y_forget_one_hot = self.encoder.transform(self.y_forget.reshape(-1, 1))
            
            loss_test = np.array([metrics.log_loss(y_test_one_hot[i], test_preds[i]) for i in range(len(self.y_test))])
            loss_forget = np.array([metrics.log_loss(y_forget_one_hot[i], forget_preds[i]) for i in range(len(self.y_forget))])
            
            idxs = np.arange(len(self.y_test))
            random.shuffle(idxs)
            rand_idxs = idxs[:m]
            
            attack_result = tf_attack(logits_train=forget_preds, logits_test=test_preds[rand_idxs], 
                                    loss_train=loss_forget, loss_test=loss_test[rand_idxs], 
                                    train_labels=self.y_forget, test_labels=self.y_test[rand_idxs])
        
        auc = attack_result.get_result_with_max_auc().get_auc()
        adv = attack_result.get_result_with_max_attacker_advantage().get_attacker_advantage()
        return auc, adv
    
    def _print_metrics(self, metrics_dict, experiment_type):
        """Print obtained metrics in a formatted way"""
        print(f"\n{'='*50}")
        print(f"OBTAINED METRICS - {experiment_type.upper()}")
        print(f"{'='*50}")
        print(f"{'Metric':<20} {'Mean':<12} {'Std Dev':<12}")
        print("-" * 50)
        for metric, (mean_val, std_val) in metrics_dict.items():
            print(f"{metric:<20} {mean_val:<12.4f} {std_val:<12.4f}")
        print("=" * 50)
    
    def _save_results(self, filename, metrics_dict):
        """Save results to CSV"""
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Metric', 'Mean', 'Standard Deviation'])
            for metric, (mean_val, std_val) in metrics_dict.items():
                writer.writerow([metric, mean_val, std_val])
    
    def run_baseline(self):
        """Run baseline experiments (training on full data and retrain on retain set)"""
        print("Running baseline experiments...")
        
        for forget_ratio in self.forget_ratios:
            print(f"\nForget ratio: {forget_ratio}")
            
            # Split data
            if self.is_tabular:
                X_retain, y_retain, self.X_forget, self.y_forget, retain_loader, forget_loader, m = self._split_forget_retain(forget_ratio)
            else:
                _, _, _, _, retain_loader, forget_loader, m = self._split_forget_retain(forget_ratio)
            
            # Baseline: Train on full data
            train_accs, test_accs, mia_aucs, mia_advs, runtimes = [], [], [], [], []
            
            for r in range(self.n_repeat):
                print(f"  Baseline experiment {r+1}/{self.n_repeat}")
                torch.cuda.empty_cache()
                model = copy.deepcopy(self.initial_model)
                
                t0 = time.time()
                if self.model_type in ['mlp', 'densenet']:
                    optimizer = optim.Adam(model.parameters(), lr=self.lr) if self.model_type == 'mlp' else optim.SGD(model.parameters(), lr=self.lr, momentum=0.9)
                    if self.is_tabular:
                        model = train_model(model, self.train_loader, self.test_loader, self.criterion, optimizer, 
                                          self.max_epochs, device=self.device, verbose_epoch=int(self.max_epochs/10) + 1)
                    else:
                        model = train_model(model, DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True), 
                                          DataLoader(self.testset, batch_size=self.batch_size, shuffle=False), 
                                          self.criterion, optimizer, self.max_epochs, device=self.device)
                else:
                    model.fit(self.X_train, self.y_train)
                
                t1 = time.time()
                runtimes.append(t1 - t0)
                
                # Evaluate
                train_acc, test_acc = self._evaluate_model(model, self.model_type != 'xgboost')
                train_accs.append(100.0 * train_acc)
                test_accs.append(100.0 * test_acc)
                
                # MIA
                auc, adv = self._compute_mia(model, forget_loader, m, self.model_type != 'xgboost')
                mia_aucs.append(100.0 * auc)
                mia_advs.append(100.0 * adv)
                
                # Print individual run results
                print(f"    Run {r+1}: Train Acc={train_acc*100:.2f}%, Test Acc={test_acc*100:.2f}%, MIA AUC={auc*100:.2f}%, MIA Adv={adv*100:.2f}%, Time={t1-t0:.2f}s")
            
            # Save baseline results
            metrics = {
                'Training Time': (np.mean(runtimes), np.std(runtimes)),
                'Train Accuracy': (np.mean(train_accs), np.std(train_accs)),
                'Test Accuracy': (np.mean(test_accs), np.std(test_accs)),
                'MIA AUC': (np.mean(mia_aucs), np.std(mia_aucs)),
                'MIA Advantage': (np.mean(mia_advs), np.std(mia_advs))
            }
            
            # Print metrics before saving
            self._print_metrics(metrics, f"Baseline (forget_ratio={forget_ratio})")
            
            filename = f'results/{self.dataset}/{self.model_type}_m_d_fr={forget_ratio}.csv'
            self._save_results(filename, metrics)
            print(f"Baseline results saved to {filename}")
            
            # Retrain baseline
            train_accs, test_accs, mia_aucs, mia_advs, runtimes = [], [], [], [], []
            
            for r in range(self.n_repeat):
                print(f"  Retrain experiment {r+1}/{self.n_repeat}")
                torch.cuda.empty_cache()
                model = copy.deepcopy(self.initial_model)
                
                t0 = time.time()
                if self.model_type in ['mlp', 'densenet']:
                    optimizer = optim.Adam(model.parameters(), lr=self.lr) if self.model_type == 'mlp' else optim.SGD(model.parameters(), lr=self.lr, momentum=0.9)
                    model = train_model(model, retain_loader, self.test_loader if self.is_tabular else DataLoader(self.testset, batch_size=self.batch_size, shuffle=False), 
                                      self.criterion, optimizer, self.max_epochs, device=self.device)
                else:
                    model.fit(X_retain, y_retain)
                
                t1 = time.time()
                runtimes.append(t1 - t0)
                
                # Evaluate retain/forget performance
                if self.model_type in ['mlp', 'densenet']:
                    if self.dataset == 'credit':
                        retain_acc = auc_score(model, retain_loader)
                        forget_acc = auc_score(model, forget_loader)
                        test_acc = auc_score(model, self.test_loader)
                    else:
                        retain_acc = accuracy(model, retain_loader)
                        forget_acc = accuracy(model, forget_loader)
                        test_acc = accuracy(model, self.test_loader if self.is_tabular else DataLoader(self.testset, batch_size=self.batch_size, shuffle=False))
                else:
                    if self.dataset == 'credit':
                        retain_acc = roc_auc_score(y_retain, model.predict_proba(X_retain)[:, 1])
                        forget_acc = roc_auc_score(self.y_forget, model.predict_proba(self.X_forget)[:, 1])
                        test_acc = roc_auc_score(self.y_test, model.predict_proba(self.X_test)[:, 1])
                    else:
                        retain_acc = accuracy_score(y_retain, model.predict(X_retain))
                        forget_acc = accuracy_score(self.y_forget, model.predict(self.X_forget))
                        test_acc = accuracy_score(self.y_test, model.predict(self.X_test))
                
                train_accs.append(100.0 * retain_acc)  # Store as "retain accuracy"
                test_accs.append(100.0 * test_acc)
                
                # MIA
                auc, adv = self._compute_mia(model, forget_loader, m, self.model_type != 'xgboost')
                mia_aucs.append(100.0 * auc)
                mia_advs.append(100.0 * adv)
                
                # Print individual run results
                print(f"    Run {r+1}: Retain Acc={retain_acc*100:.2f}%, Forget Acc={forget_acc*100:.2f}%, Test Acc={test_acc*100:.2f}%, MIA AUC={auc*100:.2f}%, MIA Adv={adv*100:.2f}%, Time={t1-t0:.2f}s")
            
            # Save retrain results
            metrics = {
                'Retraining Time': (np.mean(runtimes), np.std(runtimes)),
                'Retain Accuracy': (np.mean(train_accs), np.std(train_accs)),
                'Forget Accuracy': (np.mean([100.0 * forget_acc]), np.std([0])),  # Single value
                'Test Accuracy': (np.mean(test_accs), np.std(test_accs)),
                'MIA AUC': (np.mean(mia_aucs), np.std(mia_aucs)),
                'MIA Advantage': (np.mean(mia_advs), np.std(mia_advs))
            }
            
            # Print metrics before saving
            self._print_metrics(metrics, f"Retrain (forget_ratio={forget_ratio})")
            
            filename = f'results/{self.dataset}/{self.model_type}_mret_dret_fr={forget_ratio}.csv'
            self._save_results(filename, metrics)
            print(f"Retrain results saved to {filename}")

def main():
    parser = argparse.ArgumentParser(description='Unified Privacy Benchmark')
    parser.add_argument('--dataset', choices=['adult', 'credit', 'heart', 'cifar10'], required=True)
    parser.add_argument('--model', choices=['mlp', 'xgboost', 'densenet'], required=True)
    parser.add_argument('--forget_ratios', nargs='+', type=float, default=[0.05])
    parser.add_argument('--n_repeat', type=int, default=1)
    parser.add_argument('--max_epochs', type=int, default=10)
    
    args = parser.parse_args()
    
    # Validate model-dataset combinations
    valid_combinations = {
        'adult': ['mlp', 'xgboost'],
        'credit': ['mlp', 'xgboost'],
        'heart': ['mlp', 'xgboost'],
        'cifar10': ['densenet']
    }
    
    if args.model not in valid_combinations[args.dataset]:
        print(f"Error: {args.model} is not supported for {args.dataset} dataset")
        print(f"Valid models for {args.dataset}: {valid_combinations[args.dataset]}")
        return
    
    # Initialize benchmark
    benchmark = PrivacyBenchmark(
        dataset=args.dataset,
        model_type=args.model,
        forget_ratios=args.forget_ratios,
        n_repeat=args.n_repeat,
        max_epochs=args.max_epochs
    )
    
    # Run baseline experiments
    benchmark.run_baseline()
   
    print("Benchmark completed!")

if __name__ == "__main__":
    main()