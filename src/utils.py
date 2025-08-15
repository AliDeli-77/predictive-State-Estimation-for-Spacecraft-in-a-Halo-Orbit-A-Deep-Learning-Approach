import numpy as np
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / 'results'

def load_data(data_dir='data'):
    data_dir = Path(data_dir)
    x_train = np.load(data_dir / 'x_train.npy')
    y_train = np.load(data_dir / 'y_train.npy')
    return x_train, y_train

def save_results(true_data, pred_data, save_dir=RESULTS_DIR):
    save_dir.mkdir(exist_ok=True)
    np.save(save_dir / 'true_values.npy', true_data)
    np.save(save_dir / 'predicted_values.npy', pred_data)