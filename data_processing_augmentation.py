import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

def get_augmentations(config):
    """Retorna transformações de augmentação baseadas na configuração"""
    base_transforms = [
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), p=0.5),
        A.RandomGamma(gamma_limit=(80,120), p=0.5),
        A.HueSaturationValue(
            hue_shift_limit=config['hue_shift'],
            sat_shift_limit=config['sat_shift'],
            val_shift_limit=config['val_shift'],
            p=config['hsv_prob']
        ),
        A.ElasticTransform(
            alpha=1,
            sigma=50,
            alpha_affine=50,
            p=0.25
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ]
    
    # Augmentação específica para tons de pele latinos
    if config['ethnic_specific']:
        base_transforms.insert(0, 
            A.RGBShift(
                r_shift_limit=15,
                g_shift_limit=10,
                b_shift_limit=5,
                p=0.7
            )
        )
    
    return A.Compose(base_transforms)

def apply_ethnic_augmentation(dataset, config):
    """Aplica augmentação específica para tons de pele latinos"""
    latin_indices = [i for i, (_, metadata) in enumerate(dataset) 
                   if metadata['ethnicity'] == 'latin']
    
    latin_augmentation = A.Compose([
        A.RGBShift(
            r_shift_limit=20,
            g_shift_limit=15,
            b_shift_limit=10,
            p=1.0
        ),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=40,
            val_shift_limit=20,
            p=0.8
        )
    ])
    
    augmented_samples = []
    for idx in latin_indices:
        image, label = dataset[idx]
        augmented = latin_augmentation(image=image)['image']
        augmented_samples.append((augmented, label))
    
    dataset += augmented_samples
    return dataset