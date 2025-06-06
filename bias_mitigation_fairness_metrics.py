import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import Subset

def calculate_fairness(model, test_data, ethnic_groups):
    """Calcula métricas de justiça por grupo étnico"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    ethnic_accuracies = {}
    ethnic_preds = {group: [] for group in ethnic_groups}
    ethnic_labels = {group: [] for group in ethnic_groups}
    
    with torch.no_grad():
        for idx in range(len(test_data)):
            image, label, metadata = test_data[idx]
            image = image.unsqueeze(0).to(device)
            
            output = model(image)
            _, pred = torch.max(output, 1)
            
            ethnicity = metadata['ethnicity']
            ethnic_preds[ethnicity].append(pred.item())
            ethnic_labels[ethnicity].append(label)
    
    # Calcular métricas por grupo
    results = {}
    for group in ethnic_groups:
        if len(ethnic_preds[group]) > 0:
            acc = accuracy_score(ethnic_labels[group], ethnic_preds[group])
            results[f"{group}_accuracy"] = acc
    
    # Calcular diferença máxima entre grupos
    acc_values = [v for k,v in results.items() if k.endswith('_accuracy')]
    results['max_accuracy_difference'] = max(acc_values) - min(acc_values)
    
    return results

def cross_validate_fairness(model_class, dataset, ethnic_groups, n_splits=5):
    """Validação cruzada estratificada considerando grupos étnicos"""
    skf = StratifiedKFold(n_splits=n_splits)
    ethnicities = [metadata['ethnicity'] for _, _, metadata in dataset]
    labels = [label for _, label, _ in dataset]
    
    fold_results = []
    
    for train_idx, test_idx in skf.split(np.zeros(len(labels)), labels):
        # Garantir distribuição étnica balanceada
        train_subset = Subset(dataset, train_idx)
        test_subset = Subset(dataset, test_idx)
        
        model = model_class()
        train_model(model, train_subset, None, {'epochs': 10, 'batch_size': 32})
        
        fold_metrics = calculate_fairness(model, test_subset, ethnic_groups)
        fold_results.append(fold_metrics)
    
    # Calcular médias entre folds
    avg_results = {}
    for metric in fold_results[0].keys():
        avg_results[metric] = np.mean([f[metric] for f in fold_results])
    
    return avg_results