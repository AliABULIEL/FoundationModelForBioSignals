#!/usr/bin/env python
"""
Run TabPFN v2 for IOH (Intraoperative Hypotension) Prediction
Uses in-context learning with patient-aware splits
"""

import os
import sys
import yaml
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Tuple, List
from sklearn.metrics import classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data import VitalDBDataset
from models.tabpfn_foundation import TabPFNFoundation


def load_config(config_path: str = 'configs/tabpfn_vitaldb.yaml') -> Dict:
    """Load configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def prepare_data(config: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int], List[int]]:
    """
    Prepare tabular data with patient-aware splits
    
    Returns:
        X_context, y_context, X_query, y_query, context_patients, query_patients
    """
    # Create dataset in tabular mode
    dataset = VitalDBDataset(
        mode='tabular',
        feature_set=config['dataset']['feature_set'],
        window_sec=config['dataset']['window_sec'],
        overlap=config['dataset']['overlap'],
        target_task=config['task']['target'],
        horizon_min=config['task']['horizon_min'],
        split='train',
        cache_dir=config['dataset']['cache_dir'],
        feature_cache_dir=config['dataset']['feature_cache_dir']
    )
    
    # Collect all data
    all_features = []
    all_labels = []
    all_patients = []
    
    # Sample data (limit for initial testing)
    max_samples = config['training']['context_size'] + config['training']['query_size']
    
    for i in range(min(len(dataset), max_samples * 2)):
        features, label, context = dataset[i]
        
        # Convert to numpy
        if isinstance(features, torch.Tensor):
            features = features.numpy()
        if isinstance(label, torch.Tensor):
            label = label.numpy().squeeze()
        
        all_features.append(features)
        all_labels.append(label)
        all_patients.append(context['patient_id'])
    
    # Convert to arrays
    X = np.array(all_features)
    y = np.array(all_labels)
    patients = np.array(all_patients)
    
    # Remove NaN labels
    valid_mask = ~np.isnan(y)
    X = X[valid_mask]
    y = y[valid_mask]
    patients = patients[valid_mask]
    
    # Patient-aware split
    unique_patients = np.unique(patients)
    np.random.shuffle(unique_patients)
    
    # Split patients (not samples)
    n_context_patients = int(len(unique_patients) * 0.7)
    context_patients = unique_patients[:n_context_patients]
    query_patients = unique_patients[n_context_patients:]
    
    # Get samples for each set
    context_mask = np.isin(patients, context_patients)
    query_mask = np.isin(patients, query_patients)
    
    X_context = X[context_mask]
    y_context = y[context_mask]
    context_patient_ids = patients[context_mask]
    
    X_query = X[query_mask]
    y_query = y[query_mask]
    query_patient_ids = patients[query_mask]
    
    # Limit to configured sizes
    context_size = min(config['training']['context_size'], len(X_context))
    query_size = min(config['training']['query_size'], len(X_query))
    
    # Class balancing for context (if enabled)
    if config['training']['class_balance'] and len(np.unique(y_context)) == 2:
        # Balance positive and negative samples
        pos_idx = np.where(y_context == 1)[0]
        neg_idx = np.where(y_context == 0)[0]
        
        n_samples_per_class = context_size // 2
        
        # Sample from each class
        if len(pos_idx) > n_samples_per_class:
            pos_idx = np.random.choice(pos_idx, n_samples_per_class, replace=False)
        if len(neg_idx) > n_samples_per_class:
            neg_idx = np.random.choice(neg_idx, n_samples_per_class, replace=False)
        
        # Combine indices
        balanced_idx = np.concatenate([pos_idx, neg_idx])
        np.random.shuffle(balanced_idx)
        
        X_context = X_context[balanced_idx]
        y_context = y_context[balanced_idx]
        context_patient_ids = context_patient_ids[balanced_idx]
    else:
        # Random sampling
        if len(X_context) > context_size:
            idx = np.random.choice(len(X_context), context_size, replace=False)
            X_context = X_context[idx]
            y_context = y_context[idx]
            context_patient_ids = context_patient_ids[idx]
    
    # Sample query set
    if len(X_query) > query_size:
        idx = np.random.choice(len(X_query), query_size, replace=False)
        X_query = X_query[idx]
        y_query = y_query[idx]
        query_patient_ids = query_patient_ids[idx]
    
    return X_context, y_context, X_query, y_query, context_patient_ids.tolist(), query_patient_ids.tolist()


def main():
    """Main execution function"""
    
    # Load configuration
    config = load_config()
    
    # Set up output directory
    output_dir = Path(config['logging']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data
    X_context, y_context, X_query, y_query, context_patients, query_patients = prepare_data(config)
    
    # Data statistics (no printing yet as per requirements)
    context_stats = {
        'n_samples': len(X_context),
        'n_features': X_context.shape[1],
        'n_patients': len(np.unique(context_patients)),
        'class_distribution': dict(zip(*np.unique(y_context, return_counts=True)))
    }
    
    query_stats = {
        'n_samples': len(X_query),
        'n_patients': len(np.unique(query_patients)),
        'class_distribution': dict(zip(*np.unique(y_query, return_counts=True)))
    }
    
    # Initialize TabPFN v2 model
    model = TabPFNFoundation(
        task_type=config['model']['task_type'],
        device=config['model']['device'],
        N_ensemble_configurations=config['model']['N_ensemble_configurations'],
        max_features=config['model']['max_features'],
        use_auto_optimizer=config['model']['use_auto_optimizer']
    )
    
    # Fit context (in-context learning)
    model.fit_context(X_context, y_context)
    
    # Make predictions
    y_pred = model.predict(X_query)
    y_proba = model.predict_proba(X_query)
    
    # Evaluate
    metrics = model.evaluate(X_query, y_query)
    
    # Additional metrics
    report = classification_report(y_query, y_pred, output_dict=True)
    cm = confusion_matrix(y_query, y_pred)
    
    # Prepare results
    results = {
        'config': config,
        'context_stats': context_stats,
        'query_stats': query_stats,
        'metrics': metrics,
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }
    
    # Save results (but don't print yet)
    if config['logging']['save_predictions']:
        np.savez_compressed(
            output_dir / 'predictions.npz',
            y_true=y_query,
            y_pred=y_pred,
            y_proba=y_proba,
            query_patients=query_patients
        )
    
    if config['logging']['save_context']:
        model.save_context(str(output_dir / 'context.npz'))
    
    # Save results summary
    with open(output_dir / 'results.yaml', 'w') as f:
        yaml.dump(results, f, default_flow_style=False)
    
    return results


if __name__ == '__main__':
    # Run without execution/printing as per requirements
    results = main()
