import os
import argparse
import logging
import yaml
import torch
import numpy as np
import mlflow
import json
from typing import Dict, Any

import ray
from ray import tune
from ray.tune.integration.mlflow import MLflowLoggerCallback
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch

from utils import setup_logger, load_movielens_data
from models import MatrixFactorizationModel, NeuralCollaborativeFiltering
from train import MovieLensDataset, train_epoch, evaluate

logger = setup_logger(__name__)

def train_with_params(config: Dict[str, Any], checkpoint_dir=None):
    """
    Training function for Ray Tune hyperparameter optimization.
    This is called for each trial with different hyperparameters.
    
    Args:
        config: Hyperparameter configuration for this trial
        checkpoint_dir: Directory for checkpoints
    """
    # Set random seeds for reproducibility
    torch.manual_seed(config['random_seed'])
    np.random.seed(config['random_seed'])
    
    # Load data
    data_path = config.get('data_path', os.path.join("..", "data", "movielens", "ml-latest-small"))
    _, train_data, test_data = load_movielens_data(
        data_path=data_path,
        train_ratio=config.get('train_test_split', 0.8)
    )
    
    # Create datasets
    train_dataset = MovieLensDataset(train_data)
    val_dataset = MovieLensDataset(
        test_data, 
        user_mapping=train_dataset.user_mapping,
        item_mapping=train_dataset.item_mapping,
        train=False
    )
    
    metadata = train_dataset.get_metadata()
    num_users = metadata['num_users']
    num_items = metadata['num_items']
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=2
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2
    )
    
    # Initialize model based on configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if config['model_type'] == 'mf':
        model = MatrixFactorizationModel(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=config['embedding_dim']
        )
    else:  # 'ncf'
        model = NeuralCollaborativeFiltering(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=config['embedding_dim'],
            hidden_dims=config['hidden_dims']
        )
    
    model = model.to(device)
    
    # Initialize loss and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Load checkpoint if available
    if checkpoint_dir:
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
        model_state, optimizer_state = torch.load(checkpoint_path)
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
    
    # Training loop for hyperparameter search
    max_epochs = config.get('num_epochs', 10)
    best_val_rmse = float('inf')
    
    for epoch in range(max_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, criterion, device)
        val_rmse = val_metrics['val_rmse']
        
        # Save checkpoint
        if checkpoint_dir:
            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
            torch.save(
                (model.state_dict(), optimizer.state_dict()),
                checkpoint_path
            )
        
        # Report metrics to Ray Tune
        tune.report(
            epoch=epoch,
            train_loss=train_loss,
            val_rmse=val_rmse
        )
        
        # Track best validation RMSE for reporting
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse

def run_tuning(config: Dict[str, Any] = None):
    """
    Run hyperparameter tuning using Ray Tune.
    
    Args:
        config: Base configuration for tuning
    """
    if config is None:
        config = {
            'experiment_name': 'movie_recommender_hyperparams',
            'num_samples': 10,  # Number of trials to run
            'max_epochs': 10,   # Maximum epochs per trial
            'resources_per_trial': {
                'cpu': 2,
                'gpu': 0.5  # Fractional GPUs supported by Ray
            },
            'data_path': os.path.join("..", "data", "movielens", "ml-latest-small"),
            'output_path': 'ray_results',
            'model_type': 'ncf',
            'search_space': {
                'batch_size': [128, 256, 512, 1024],
                'learning_rate': [1e-4, 3e-4, 1e-3, 3e-3],
                'embedding_dim': [32, 64, 128],
                'hidden_dims': [
                    [128, 64],
                    [256, 128, 64],
                    [512, 256, 128]
                ],
                'weight_decay': [0, 1e-5, 1e-4]
            }
        }
    
    # Initialize MLflow for tracking
    mlflow.set_experiment(config['experiment_name'])
    
    # Initialize Ray
    ray.init(
        num_cpus=config.get('num_cpus', None),
        num_gpus=config.get('num_gpus', None),
        log_to_driver=False
    )
    
    logger.info("Starting hyperparameter tuning with Ray Tune...")
    
    # Create search space for HyperOpt
    search_space = {}
    
    for param, values in config['search_space'].items():
        if param == 'hidden_dims':
            # For nested parameters like hidden_dims, we use a categorical choice
            search_space[param] = tune.choice(values)
        elif isinstance(values, list):
            search_space[param] = tune.choice(values)
        else:
            # If a range is specified instead of discrete values
            search_space[param] = tune.uniform(values[0], values[1])
    
    # Add fixed parameters to search space
    fixed_params = {
        'model_type': config['model_type'],
        'data_path': config['data_path'],
        'train_test_split': 0.8,
        'num_epochs': config['max_epochs'],
        'random_seed': 42
    }
    
    search_space.update(fixed_params)
    
    # Set up HyperOpt search algorithm
    search_algo = HyperOptSearch(
        metric="val_rmse",
        mode="min",
        points_to_evaluate=[{
            'model_type': config['model_type'],
            'batch_size': 256,
            'learning_rate': 1e-3,
            'embedding_dim': 64,
            'hidden_dims': [256, 128, 64],
            'weight_decay': 1e-5,
            'data_path': config['data_path'],
            'train_test_split': 0.8,
            'num_epochs': config['max_epochs'],
            'random_seed': 42
        }]  # Start with reasonable defaults
    )
    
    # Set up ASHA scheduler for early stopping of bad trials
    scheduler = ASHAScheduler(
        max_t=config['max_epochs'],
        grace_period=2,
        reduction_factor=2
    )
    
    # Set up MLflow logger for Ray Tune
    mlflow_callback = MLflowLoggerCallback(
        experiment_name=config['experiment_name'],
        save_artifact=True
    )
    
    # Run hyperparameter tuning
    analysis = tune.run(
        train_with_params,
        config=search_space,
        search_alg=search_algo,
        scheduler=scheduler,
        num_samples=config['num_samples'],
        resources_per_trial=config['resources_per_trial'],
        checkpoint_at_end=True,
        local_dir=config['output_path'],
        callbacks=[mlflow_callback],
        verbose=1,
        progress_reporter=tune.CLIReporter(
            metric_columns=["train_loss", "val_rmse", "epoch"]
        )
    )
    
    # Get best trial
    best_trial = analysis.get_best_trial("val_rmse", "min", "last")
    best_config = best_trial.config
    best_rmse = best_trial.last_result["val_rmse"]
    
    logger.info(f"Best trial config: {best_config}")
    logger.info(f"Best trial final validation RMSE: {best_rmse}")
    
    # Save best configuration
    output_file = os.path.join(config['output_path'], "best_config.json")
    with open(output_file, 'w') as f:
        json.dump(best_config, f, indent=2)
    
    logger.info(f"Best configuration saved to {output_file}")
    
    # Log best configuration to MLflow
    with mlflow.start_run(run_name="tuning_summary"):
        mlflow.log_params(best_config)
        mlflow.log_metrics({"best_val_rmse": best_rmse})
        mlflow.log_artifact(output_file)
    
    # Shut down Ray
    ray.shutdown()
    
    return best_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run hyperparameter tuning")
    parser.add_argument("--config", type=str, help="Path to config YAML file")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of trials to run")
    parser.add_argument("--max_epochs", type=int, default=10, help="Maximum epochs per trial")
    parser.add_argument("--model_type", type=str, choices=["mf", "ncf"], default="ncf", 
                        help="Model type: Matrix Factorization (mf) or Neural CF (ncf)")
    parser.add_argument("--output_path", type=str, default="ray_results", 
                        help="Path to save tuning results")
    
    args = parser.parse_args()
    
    if args.config:
        # Load configuration from YAML file
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Use default configuration with CLI overrides
        config = {
            'experiment_name': 'movie_recommender_hyperparams',
            'num_samples': args.num_samples,
            'max_epochs': args.max_epochs,
            'resources_per_trial': {
                'cpu': 2,
                'gpu': 0.5  # Fractional GPUs supported by Ray
            },
            'data_path': os.path.join("..", "data", "movielens", "ml-latest-small"),
            'output_path': args.output_path,
            'model_type': args.model_type,
            'search_space': {
                'batch_size': [128, 256, 512, 1024],
                'learning_rate': [1e-4, 3e-4, 1e-3, 3e-3],
                'embedding_dim': [32, 64, 128],
                'hidden_dims': [
                    [128, 64],
                    [256, 128, 64],
                    [512, 256, 128]
                ],
                'weight_decay': [0, 1e-5, 1e-4]
            }
        }
    
    # Run hyperparameter tuning
    best_config = run_tuning(config)
    
    logger.info("Hyperparameter tuning completed successfully.")