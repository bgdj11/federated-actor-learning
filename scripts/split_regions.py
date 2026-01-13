import numpy as np
import os

INPUT_FILE = 'eurosat_features.npz'
OUTPUT_DIR = 'dataset'

# 0: AnnualCrop, 1: Forest, 2: HerbaceousVegetation, 3: Highway,
# 4: Industrial, 5: Pasture, 6: PermanentCrop, 7: Residential,
# 8: River, 9: SeaLake

REGIONS = {
    'A': {
        'shift': 'haze',
        'classes': [1, 2, 5, 6], # rural
        'description': 'Ruralni region sa maglom/haze'
    },
    'B': {
        'shift': 'color',
        'classes': [3, 4, 7], # urban
        'description': 'Urbani region sa color shift-om (drugi senzor)'
    },
    'C': {
        'shift': 'contrast',
        'classes': [0, 8, 9], # water/agro
        'description': 'Vodni/agro region sa visokim kontrastom'
    },
    'D': {
        'shift': 'normal',
        'classes': list(range(10)),
        'description': 'Kontrolni region bez transformacija'
    }
}

def apply_shift(X, shift_type):
    if shift_type == 'haze':
        return X * 0.8 + np.random.normal(0, 0.05, X.shape).astype(np.float32)
    
    elif shift_type == 'color':
        shift = np.random.uniform(-0.15, 0.15, (1, X.shape[1])).astype(np.float32)
        return X + shift
    
    elif shift_type == 'contrast':
        return X * 1.2
    
    else:
        return X


data = np.load(INPUT_FILE)
X_all = data['features']
y_all = data['labels']

os.makedirs(OUTPUT_DIR, exist_ok=True)

for region_name, cfg in REGIONS.items():
    print(f"\nRegion {region_name}: {cfg['description']}")
    
    mask = np.isin(y_all, cfg['classes'])
    X_region = X_all[mask].copy()
    y_region = y_all[mask].copy()
    
    X_region = apply_shift(X_region, cfg['shift'])

    output_file = os.path.join(OUTPUT_DIR, f'region_{region_name}.npz')
    np.savez(output_file, X=X_region, y=y_region)
    print(f"Region {region_name}: {X_region.shape}")
