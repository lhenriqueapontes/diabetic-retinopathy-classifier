import argparse
from data_processing.dataset_loader import load_datasets
from data_processing.preprocessing import apply_preprocessing
from data_processing.augmentation import get_augmentations
from models.model_loader import load_model
from models.training import train_model
from bias_mitigation.ethnic_augmentation import apply_ethnic_augmentation
from bias_mitigation.fairness_metrics import calculate_fairness
import yaml

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Carregar e preparar dados
    train_ds, val_ds, test_ds = load_datasets(config['data_paths'])
    train_ds = apply_preprocessing(train_ds, config['preprocessing'])
    val_ds = apply_preprocessing(val_ds, config['preprocessing'])
    
    # Aplicar augmentação étnica específica
    if config['bias_mitigation']['use_ethnic_augmentation']:
        train_ds = apply_ethnic_augmentation(train_ds, config['bias_mitigation'])
    
    # Carregar modelo
    model = load_model(config['model'])
    
    # Treinar modelo
    history = train_model(
        model=model,
        train_data=train_ds,
        val_data=val_ds,
        params=config['training']
    )
    
    # Avaliar viés
    fairness_results = calculate_fairness(
        model=model,
        test_data=test_ds,
        ethnic_groups=config['bias_mitigation']['ethnic_groups']
    )
    
    print("Fairness Metrics:", fairness_results)

if __name__ == "__main__":
    main()