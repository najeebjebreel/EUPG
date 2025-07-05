#!/usr/bin/env python3
"""
Privacy experiments extension for k-anonymity and differential privacy.
"""

import pandas as pd
import numpy as np
import copy
import time
import torch
import torch.optim as optim
from baseline_experiments import PrivacyBenchmark, DatasetLoader
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, roc_auc_score
from utils import *
from train import *
from mdav import mdav

class PrivacyExperiments(PrivacyBenchmark):
    """Extended benchmark class for privacy experiments"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _print_phase_results(self, results, phase_name, k_or_eps=None, ft_epochs=None, forget_ratio=None):
        """Print results for a specific phase with detailed formatting"""
        print(f"\n{'='*60}")
        title = f"PHASE RESULTS - {phase_name.upper()}"
        if k_or_eps is not None:
            if 'k=' in str(k_or_eps):
                title += f" (k={k_or_eps}, "
            else:
                title += f" (eps={k_or_eps}, "
            if ft_epochs is not None:
                title += f"ft_epochs={ft_epochs}, "
            title += f"forget_ratio={forget_ratio})"
        else:
            title += f" (forget_ratio={forget_ratio})"
        print(title)
        print(f"{'='*60}")
        print(f"{'Metric':<20} {'Mean':<12} {'Std Dev':<12}")
        print("-" * 60)
        
        for key, values in results.items():
            if values:  # Check if list is not empty
                mean_val = np.mean(values)
                std_val = np.std(values)
                metric_name = key.replace('_', ' ').title()
                print(f"{metric_name:<20} {mean_val:<12.4f} {std_val:<12.4f}")
        print("=" * 60)
    
    def _print_individual_run(self, run_num, phase, **metrics):
        """Print individual run results"""
        metrics_str = ", ".join([f"{k}={v:.2f}" for k, v in metrics.items()])
        print(f"    Run {run_num} ({phase}): {metrics_str}")
    
    def run_kanonymity(self, k_values=[3, 5, 10, 20, 80], ft_epochs_list=[5, 10, 20]):
        """Run k-anonymity experiments"""
        print("Running k-anonymity experiments...")
        
        for forget_ratio in self.forget_ratios:
            print(f"\nForget ratio: {forget_ratio}")
            
            # Split data
            if self.is_tabular:
                X_retain, y_retain, self.X_forget, self.y_forget, retain_loader, forget_loader, m = self._split_forget_retain(forget_ratio)
            else:
                print("K-anonymity is not supported for CIFAR-10 dataset")
                continue  # Skip CIFAR-10 for k-anonymity
            
            for ft_epochs in ft_epochs_list:
                for k in k_values:
                    print(f"\n--- k={k}, fine-tuning epochs={ft_epochs} ---")
                    
                    # K-anonymize data
                    print("Performing k-anonymization...")
                    t0 = time.time()
                    centroids, clusters, labels, X_train_k, y_train_k = mdav(
                        copy.deepcopy(self.X_train), copy.deepcopy(self.y_train), k
                    )
                    anonymize_time = time.time() - t0
                    print(f"K-anonymization completed in {anonymize_time:.2f} seconds")
                    
                    train_dataset_k = TensorDataset(
                        torch.tensor(X_train_k, dtype=torch.float32), 
                        torch.tensor(y_train_k, dtype=torch.int64)
                    )
                    train_loader_k = DataLoader(train_dataset_k, batch_size=self.batch_size, shuffle=True)
                    
                    # Run experiments
                    results = self._run_privacy_experiment(
                        train_loader_k, retain_loader, forget_loader, m, ft_epochs, anonymize_time, 
                        experiment_type="k-anonymity", k_or_eps=k
                    )
                    
                    # Print and save results
                    self._save_kanonymity_results(k, forget_ratio, ft_epochs, results)
    
    def run_differential_privacy(self, eps_values=[0.5, 2.5, 5.0, 25.0, 50.0, 100.0], ft_epochs_list=[5, 10, 20]):
        """Run differential privacy experiments"""
        print("Running differential privacy experiments...")
        
        for forget_ratio in self.forget_ratios:
            print(f"\nForget ratio: {forget_ratio}")
            
            # Split data
            if self.is_tabular:
                X_retain, y_retain, self.X_forget, self.y_forget, retain_loader, forget_loader, m = self._split_forget_retain(forget_ratio)
            else:
                _, _, _, _, retain_loader, forget_loader, m = self._split_forget_retain(forget_ratio)
            
            for ft_epochs in ft_epochs_list:
                for eps in eps_values:
                    print(f"\n--- epsilon={eps}, fine-tuning epochs={ft_epochs} ---")
                    
                    # Load DP data
                    print("Loading differentially private data...")
                    train_loader_dp = self._load_dp_data(eps)
                    if train_loader_dp is None:
                        print(f"Skipping eps={eps} due to missing data file")
                        continue
                    
                    # Run experiments
                    results = self._run_privacy_experiment(
                        train_loader_dp, retain_loader, forget_loader, m, ft_epochs, 0.0,
                        experiment_type="differential_privacy", k_or_eps=eps
                    )
                    
                    # Print and save results
                    self._save_dp_results(eps, forget_ratio, ft_epochs, results)
    
    def _load_dp_data(self, eps):
        """Load differentially private data"""
        try:
            if self.dataset == 'adult':
                dp_data = pd.read_csv(f'dp_data/adult/dp_adult_eps={eps}.csv', sep=r' *, *', engine='python', na_values='?')
                dp_data.drop(['fnlwgt', 'education'], axis=1, inplace=True)
                dp_data.dropna(inplace=True)
                dp_data["income"] = dp_data["income"].apply(lambda x: 0 if x=='<=50K' else 1)
                X_train_dp = dp_data.drop('income', axis=1)
                y_train_dp = dp_data['income'].values
                X_train_dp = self.preprocessor.fit_transform(X_train_dp)
                
            elif self.dataset == 'credit':
                dp_data = pd.read_csv(f'dp_data/GiveMeSomeCredit/dp_credit_eps={eps}.csv')
                dp_data.dropna(inplace=True)
                X_train_dp = dp_data.drop('SeriousDlqin2yrs', axis=1)
                y_train_dp = dp_data['SeriousDlqin2yrs'].values
                X_train_dp = self.preprocessor.fit_transform(X_train_dp)
                
            elif self.dataset == 'heart':
                dp_data = pd.read_csv(f'dp_data/heart/dp_heart_eps={eps}.csv')
                dp_data.dropna(inplace=True)
                X_train_dp = dp_data.drop('cardio', axis=1)
                y_train_dp = dp_data['cardio'].values
                X_train_dp = self.preprocessor.fit_transform(X_train_dp)
                
            elif self.dataset == 'cifar10':
                # For CIFAR-10, use the DP dataset loader
                dataset_dir = f'dp_data/cifar10/m16_b4/dp_cifar10_eps_{eps}'
                labels = self.trainset.targets
                from dp_data.load_dp_cifar10_dataset import DPCIFAR10Dataset
                from torchvision import transforms
                
                transform_train = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
                
                train_dataset_dp = DPCIFAR10Dataset(root_dir=dataset_dir, labels=labels, transform=transform_train)
                return DataLoader(train_dataset_dp, batch_size=self.batch_size, shuffle=True)
            
            if self.dataset != 'cifar10':
                train_dataset_dp = TensorDataset(
                    torch.tensor(X_train_dp, dtype=torch.float32), 
                    torch.tensor(y_train_dp, dtype=torch.int64)
                )
                return DataLoader(train_dataset_dp, batch_size=self.batch_size, shuffle=True)
                
        except FileNotFoundError:
            print(f"DP data file not found for eps={eps}")
            return None
    
    def _run_privacy_experiment(self, train_loader_private, retain_loader, forget_loader, m, ft_epochs, prep_time, experiment_type="", k_or_eps=None):
        """Run a complete privacy experiment with three phases"""
        results = {
            'phase1': {'times': [], 'train_accs': [], 'test_accs': [], 'mia_aucs': [], 'mia_advs': []},
            'phase2': {'times': [], 'train_accs': [], 'test_accs': [], 'mia_aucs': [], 'mia_advs': []},
            'phase3': {'times': [], 'retain_accs': [], 'forget_accs': [], 'test_accs': [], 'mia_aucs': [], 'mia_advs': []}
        }
        
        for r in range(self.n_repeat):
            print(f"\n  Experiment run {r+1}/{self.n_repeat}")
            torch.cuda.empty_cache()
            
            # Phase 1: Train on privacy-preserved data
            print("    Phase 1: Training on privacy-preserved data...")
            model_private = copy.deepcopy(self.initial_model)
            t0 = time.time()
            
            if self.model_type in ['mlp', 'densenet']:
                optimizer = optim.Adam(model_private.parameters(), lr=self.lr) if self.model_type == 'mlp' else optim.SGD(model_private.parameters(), lr=self.lr, momentum=0.9)
                epochs = self.max_epochs
                test_loader = self.test_loader if self.is_tabular else DataLoader(self.testset, batch_size=self.batch_size, shuffle=False)
                model_private = train_model(model_private, train_loader_private, test_loader, self.criterion, optimizer, 
                                          epochs, device=self.device, verbose_epoch=int(epochs/10) + 1)
            else:
                # XGBoost
                # Extract data from DataLoader for XGBoost
                X_private, y_private = self._extract_data_from_loader(train_loader_private)
                model_private.fit(X_private, y_private)
            
            phase1_time = time.time() - t0
            results['phase1']['times'].append(phase1_time)
            
            # Evaluate Phase 1
            train_acc, test_acc = self._evaluate_model(model_private, self.model_type != 'xgboost')
            results['phase1']['train_accs'].append(100.0 * train_acc)
            results['phase1']['test_accs'].append(100.0 * test_acc)
            
            auc, adv = self._compute_mia(model_private, forget_loader, m, self.model_type != 'xgboost')
            results['phase1']['mia_aucs'].append(100.0 * auc)
            results['phase1']['mia_advs'].append(100.0 * adv)
            
            # Print Phase 1 individual results
            self._print_individual_run(r+1, "Phase 1", 
                                     train_acc=train_acc*100, test_acc=test_acc*100, 
                                     mia_auc=auc*100, mia_adv=adv*100, time=phase1_time)
            
            # Phase 2: Fine-tune on original data
            print("    Phase 2: Fine-tuning on original data...")
            model_ft = copy.deepcopy(model_private)
            t0 = time.time()
            
            if self.model_type in ['mlp', 'densenet']:
                optimizer = optim.Adam(model_ft.parameters(), lr=self.lr) if self.model_type == 'mlp' else optim.SGD(model_ft.parameters(), lr=self.lr, momentum=0.9)
                test_loader = self.test_loader if self.is_tabular else DataLoader(self.testset, batch_size=self.batch_size, shuffle=False)
                train_loader = self.train_loader if self.is_tabular else DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)
                model_ft = train_model(model_ft, train_loader, test_loader, self.criterion, optimizer, 
                                     ft_epochs, device=self.device, verbose_epoch=int(self.max_epochs/10) + 1)
            else:
                # XGBoost fine-tuning
                model_ft.set_params(learning_rate=0.5, n_estimators=ft_epochs)
                model_ft.fit(self.X_train, self.y_train, xgb_model=model_private)
            
            phase2_time = time.time() - t0
            results['phase2']['times'].append(phase2_time)
            
            # Evaluate Phase 2
            train_acc, test_acc = self._evaluate_model(model_ft, self.model_type != 'xgboost')
            results['phase2']['train_accs'].append(100.0 * train_acc)
            results['phase2']['test_accs'].append(100.0 * test_acc)
            
            auc, adv = self._compute_mia(model_ft, forget_loader, m, self.model_type != 'xgboost')
            results['phase2']['mia_aucs'].append(100.0 * auc)
            results['phase2']['mia_advs'].append(100.0 * adv)
            
            # Print Phase 2 individual results
            self._print_individual_run(r+1, "Phase 2", 
                                     train_acc=train_acc*100, test_acc=test_acc*100, 
                                     mia_auc=auc*100, mia_adv=adv*100, time=phase2_time)
            
            # Phase 3: Fine-tune on retain data
            print("    Phase 3: Fine-tuning on retain data...")
            model_retain = copy.deepcopy(model_private)
            t0 = time.time()
            
            if self.model_type in ['mlp', 'densenet']:
                optimizer = optim.Adam(model_retain.parameters(), lr=self.lr) if self.model_type == 'mlp' else optim.SGD(model_retain.parameters(), lr=self.lr, momentum=0.9)
                test_loader = self.test_loader if self.is_tabular else DataLoader(self.testset, batch_size=self.batch_size, shuffle=False)
                model_retain = train_model(model_retain, retain_loader, test_loader, self.criterion, optimizer, 
                                         ft_epochs, device=self.device, verbose_epoch=int(self.max_epochs/10) + 1)
            else:
                # XGBoost fine-tuning on retain set
                X_retain, y_retain = self._extract_data_from_loader(retain_loader)
                model_retain.set_params(learning_rate=0.5, n_estimators=ft_epochs)
                model_retain.fit(X_retain, y_retain, xgb_model=model_private)
            
            phase3_time = time.time() - t0
            results['phase3']['times'].append(phase3_time)
            
            # Evaluate Phase 3 (retain/forget performance)
            if self.model_type in ['mlp', 'densenet']:
                if self.dataset == 'credit':
                    retain_acc = auc_score(model_retain, retain_loader)
                    forget_acc = auc_score(model_retain, forget_loader)
                    test_acc = auc_score(model_retain, self.test_loader if self.is_tabular else DataLoader(self.testset, batch_size=self.batch_size, shuffle=False))
                else:
                    retain_acc = accuracy(model_retain, retain_loader)
                    forget_acc = accuracy(model_retain, forget_loader)
                    test_acc = accuracy(model_retain, self.test_loader if self.is_tabular else DataLoader(self.testset, batch_size=self.batch_size, shuffle=False))
            else:
                X_retain, y_retain = self._extract_data_from_loader(retain_loader)
                X_forget, y_forget = self._extract_data_from_loader(forget_loader)
                
                if self.dataset == 'credit':
                    retain_acc = roc_auc_score(y_retain, model_retain.predict_proba(X_retain)[:, 1])
                    forget_acc = roc_auc_score(y_forget, model_retain.predict_proba(X_forget)[:, 1])
                    test_acc = roc_auc_score(self.y_test, model_retain.predict_proba(self.X_test)[:, 1])
                else:
                    retain_acc = accuracy_score(y_retain, model_retain.predict(X_retain))
                    forget_acc = accuracy_score(y_forget, model_retain.predict(X_forget))
                    test_acc = accuracy_score(self.y_test, model_retain.predict(self.X_test))
            
            results['phase3']['retain_accs'].append(100.0 * retain_acc)
            results['phase3']['forget_accs'].append(100.0 * forget_acc)
            results['phase3']['test_accs'].append(100.0 * test_acc)
            
            auc, adv = self._compute_mia(model_retain, forget_loader, m, self.model_type != 'xgboost')
            results['phase3']['mia_aucs'].append(100.0 * auc)
            results['phase3']['mia_advs'].append(100.0 * adv)
            
            # Print Phase 3 individual results
            self._print_individual_run(r+1, "Phase 3", 
                                     retain_acc=retain_acc*100, forget_acc=forget_acc*100, 
                                     test_acc=test_acc*100, mia_auc=auc*100, mia_adv=adv*100, time=phase3_time)
        
        # Add preparation time to phase 1
        for i in range(len(results['phase1']['times'])):
            results['phase1']['times'][i] += prep_time
        
        # Print summary results for all phases
        print(f"\n{'='*80}")
        print(f"SUMMARY RESULTS - {experiment_type.upper()} EXPERIMENT")
        if k_or_eps is not None:
            if experiment_type == "k-anonymity":
                print(f"k={k_or_eps}, ft_epochs={ft_epochs}")
            else:
                print(f"eps={k_or_eps}, ft_epochs={ft_epochs}")
        print(f"{'='*80}")
        
        # Phase 1 Summary
        self._print_phase_results(results['phase1'], f"Phase 1 - Train on {experiment_type} data", 
                                k_or_eps, ft_epochs, forget_ratio=self.forget_ratios[0])
        
        # Phase 2 Summary  
        self._print_phase_results(results['phase2'], "Phase 2 - Fine-tune on original data", 
                                k_or_eps, ft_epochs, forget_ratio=self.forget_ratios[0])
        
        # Phase 3 Summary
        self._print_phase_results(results['phase3'], "Phase 3 - Fine-tune on retain data", 
                                k_or_eps, ft_epochs, forget_ratio=self.forget_ratios[0])
        
        return results
    
    def _extract_data_from_loader(self, data_loader):
        """Extract X, y arrays from PyTorch DataLoader for XGBoost"""
        X_list, y_list = [], []
        for batch_x, batch_y in data_loader:
            X_list.append(batch_x.numpy())
            y_list.append(batch_y.numpy())
        return np.vstack(X_list), np.hstack(y_list)
    
    def _save_kanonymity_results(self, k, forget_ratio, ft_epochs, results):
        """Save k-anonymity results to CSV files"""
        base_path = f'results/{self.dataset}'
        
        # Phase 1: Training on k-anonymous data
        metrics = {
            'Anonymizing Time': (np.mean(results['phase1']['times']), np.std(results['phase1']['times'])),
            'Train Accuracy': (np.mean(results['phase1']['train_accs']), np.std(results['phase1']['train_accs'])),
            'Test Accuracy': (np.mean(results['phase1']['test_accs']), np.std(results['phase1']['test_accs'])),
            'MIA AUC': (np.mean(results['phase1']['mia_aucs']), np.std(results['phase1']['mia_aucs'])),
            'MIA Advantage': (np.mean(results['phase1']['mia_advs']), np.std(results['phase1']['mia_advs']))
        }
        
        print(f"\nSaving Phase 1 k-anonymity results (k={k})...")
        self._print_metrics(metrics, f"K-Anonymity Phase 1 (k={k}, forget_ratio={forget_ratio})")
        
        filename = f'{base_path}/{self.model_type}_mk={k}_dk_fr={forget_ratio}.csv'
        self._save_results(filename, metrics)
        print(f"Phase 1 results saved to {filename}")
        
        # Phase 2: Fine-tuning on original data
        metrics = {
            'Training Time': (np.mean(results['phase2']['times']), np.std(results['phase2']['times'])),
            'Train Accuracy': (np.mean(results['phase2']['train_accs']), np.std(results['phase2']['train_accs'])),
            'Test Accuracy': (np.mean(results['phase2']['test_accs']), np.std(results['phase2']['test_accs'])),
            'MIA AUC': (np.mean(results['phase2']['mia_aucs']), np.std(results['phase2']['mia_aucs'])),
            'MIA Advantage': (np.mean(results['phase2']['mia_advs']), np.std(results['phase2']['mia_advs']))
        }
        
        print(f"\nSaving Phase 2 k-anonymity results (k={k})...")
        self._print_metrics(metrics, f"K-Anonymity Phase 2 (k={k}, ft_epochs={ft_epochs}, forget_ratio={forget_ratio})")
        
        filename = f'{base_path}/{self.model_type}_mk={k}_d_fr={forget_ratio}_epochs={ft_epochs}.csv'
        self._save_results(filename, metrics)
        print(f"Phase 2 results saved to {filename}")
        
        # Phase 3: Fine-tuning on retain data
        metrics = {
            'Training Time': (np.mean(results['phase3']['times']), np.std(results['phase3']['times'])),
            'Retain Accuracy': (np.mean(results['phase3']['retain_accs']), np.std(results['phase3']['retain_accs'])),
            'Forget Accuracy': (np.mean(results['phase3']['forget_accs']), np.std(results['phase3']['forget_accs'])),
            'Test Accuracy': (np.mean(results['phase3']['test_accs']), np.std(results['phase3']['test_accs'])),
            'MIA AUC': (np.mean(results['phase3']['mia_aucs']), np.std(results['phase3']['mia_aucs'])),
            'MIA Advantage': (np.mean(results['phase3']['mia_advs']), np.std(results['phase3']['mia_advs']))
        }
        
        print(f"\nSaving Phase 3 k-anonymity results (k={k})...")
        self._print_metrics(metrics, f"K-Anonymity Phase 3 (k={k}, ft_epochs={ft_epochs}, forget_ratio={forget_ratio})")
        
        filename = f'{base_path}/{self.model_type}_mk={k}_dret_fr={forget_ratio}_epochs={ft_epochs}.csv'
        self._save_results(filename, metrics)
        print(f"Phase 3 results saved to {filename}")
    
    def _save_dp_results(self, eps, forget_ratio, ft_epochs, results):
        """Save differential privacy results to CSV files"""
        base_path = f'results/{self.dataset}'
        
        # Phase 1: Training on DP data
        metrics = {
            'Training Time': (np.mean(results['phase1']['times']), np.std(results['phase1']['times'])),
            'Train Accuracy': (np.mean(results['phase1']['train_accs']), np.std(results['phase1']['train_accs'])),
            'Test Accuracy': (np.mean(results['phase1']['test_accs']), np.std(results['phase1']['test_accs'])),
            'MIA AUC': (np.mean(results['phase1']['mia_aucs']), np.std(results['phase1']['mia_aucs'])),
            'MIA Advantage': (np.mean(results['phase1']['mia_advs']), np.std(results['phase1']['mia_advs']))
        }
        
        print(f"\nSaving Phase 1 DP results (eps={eps})...")
        self._print_metrics(metrics, f"Differential Privacy Phase 1 (eps={eps}, forget_ratio={forget_ratio})")
        
        filename = f'{base_path}/{self.model_type}_mdp_eps={eps}_fr={forget_ratio}.csv'
        self._save_results(filename, metrics)
        print(f"Phase 1 results saved to {filename}")
        
        # Phase 2: Fine-tuning on original data
        metrics = {
            'Training Time': (np.mean(results['phase2']['times']), np.std(results['phase2']['times'])),
            'Train Accuracy': (np.mean(results['phase2']['train_accs']), np.std(results['phase2']['train_accs'])),
            'Test Accuracy': (np.mean(results['phase2']['test_accs']), np.std(results['phase2']['test_accs'])),
            'MIA AUC': (np.mean(results['phase2']['mia_aucs']), np.std(results['phase2']['mia_aucs'])),
            'MIA Advantage': (np.mean(results['phase2']['mia_advs']), np.std(results['phase2']['mia_advs']))
        }
        
        print(f"\nSaving Phase 2 DP results (eps={eps})...")
        self._print_metrics(metrics, f"Differential Privacy Phase 2 (eps={eps}, ft_epochs={ft_epochs}, forget_ratio={forget_ratio})")
        
        filename = f'{base_path}/{self.model_type}_mdpd_eps={eps}_fr={forget_ratio}_epochs={ft_epochs}.csv'
        self._save_results(filename, metrics)
        print(f"Phase 2 results saved to {filename}")
        
        # Phase 3: Fine-tuning on retain data
        metrics = {
            'Training Time': (np.mean(results['phase3']['times']), np.std(results['phase3']['times'])),
            'Retain Accuracy': (np.mean(results['phase3']['retain_accs']), np.std(results['phase3']['retain_accs'])),
            'Forget Accuracy': (np.mean(results['phase3']['forget_accs']), np.std(results['phase3']['forget_accs'])),
            'Test Accuracy': (np.mean(results['phase3']['test_accs']), np.std(results['phase3']['test_accs'])),
            'MIA AUC': (np.mean(results['phase3']['mia_aucs']), np.std(results['phase3']['mia_aucs'])),
            'MIA Advantage': (np.mean(results['phase3']['mia_advs']), np.std(results['phase3']['mia_advs']))
        }
        
        print(f"\nSaving Phase 3 DP results (eps={eps})...")
        self._print_metrics(metrics, f"Differential Privacy Phase 3 (eps={eps}, ft_epochs={ft_epochs}, forget_ratio={forget_ratio})")
        
        filename = f'{base_path}/{self.model_type}_mdpret_eps={eps}_fr={forget_ratio}_epochs={ft_epochs}.csv'
        self._save_results(filename, metrics)
        print(f"Phase 3 results saved to {filename}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Privacy Experiments')
    parser.add_argument('--dataset', choices=['adult', 'credit', 'heart', 'cifar10'], required=True)
    parser.add_argument('--model', choices=['mlp', 'xgboost', 'densenet'], required=True)
    parser.add_argument('--forget_ratios', nargs='+', type=float, default=[0.05])
    parser.add_argument('--n_repeat', type=int, default=3)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--experiments', nargs='+', choices=['baseline', 'kanonymity', 'dp'], default=['baseline'])
    parser.add_argument('--k_values', nargs='+', type=int, default=[10])
    parser.add_argument('--eps_values', nargs='+', type=float, default=[2.5])
    parser.add_argument('--ft_epochs', nargs='+', type=int, default=[5])
    
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
    
    # Initialize experiments
    experiments = PrivacyExperiments(
        dataset=args.dataset,
        model_type=args.model,
        forget_ratios=args.forget_ratios,
        n_repeat=args.n_repeat,
        max_epochs=args.max_epochs
    )
    
    # Run experiments
    if 'baseline' in args.experiments:
        print("\n" + "="*80)
        print("STARTING BASELINE EXPERIMENTS")
        print("="*80)
        experiments.run_baseline()
    
    if 'kanonymity' in args.experiments:
        print("\n" + "="*80)
        print("STARTING K-ANONYMITY EXPERIMENTS")
        print("="*80)
        experiments.run_kanonymity(k_values=args.k_values, ft_epochs_list=args.ft_epochs)
    
    if 'dp' in args.experiments:
        print("\n" + "="*80)
        print("STARTING DIFFERENTIAL PRIVACY EXPERIMENTS")
        print("="*80)
        experiments.run_differential_privacy(eps_values=args.eps_values, ft_epochs_list=args.ft_epochs)
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("="*80)

if __name__ == "__main__":
    main()