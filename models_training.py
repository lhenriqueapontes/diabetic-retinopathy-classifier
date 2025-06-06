import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score, f1_score
import numpy as np

def train_model(model, train_data, val_data, params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=params['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',
        patience=params['scheduler_patience']
    )
    
    best_kappa = 0
    early_stop_counter = 0
    
    for epoch in range(params['epochs']):
        # Treinamento
        model.train()
        train_loss = 0
        all_preds = []
        all_labels = []
        
        for images, labels in tqdm(DataLoader(train_data, batch_size=params['batch_size'])):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        train_loss /= len(train_data)
        train_kappa = cohen_kappa_score(all_labels, all_preds)
        train_f1 = f1_score(all_labels, all_preds, average='weighted')
        
        # Validação
        val_metrics = evaluate_model(model, val_data, device, params['batch_size'])
        
        print(f"Epoch {epoch+1}/{params['epochs']}")
        print(f"Train Loss: {train_loss:.4f} | Kappa: {train_kappa:.4f} | F1: {train_f1:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f} | Kappa: {val_metrics['kappa']:.4f} | F1: {val_metrics['f1']:.4f}")
        
        # Early stopping e scheduler
        scheduler.step(val_metrics['kappa'])
        
        if val_metrics['kappa'] > best_kappa:
            best_kappa = val_metrics['kappa']
            early_stop_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            early_stop_counter += 1
            if early_stop_counter >= params['early_stopping_patience']:
                print("Early stopping triggered")
                break
    
    return {
        'best_kappa': best_kappa,
        'best_model': 'best_model.pth'
    }

def evaluate_model(model, dataset, device, batch_size):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in DataLoader(dataset, batch_size=batch_size):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    metrics = {
        'loss': total_loss / len(dataset),
        'kappa': cohen_kappa_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds, average='weighted'),
        'accuracy': sum(1 for x,y in zip(all_preds, all_labels) if x == y) / len(all_labels)
    }
    
    return metrics