"""
Hyperparameter tuning for movie recommendation models using Ray Tune.
"""

import os
import logging
import time
from functools import partial
from typing import Dict, List, Any, Optional, Tuple, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import ray
from ray import tune
from ray.air import session
from ray.tune.schedulers import ASHAScheduler
from ray.air.config import ScalingConfig

from models import get_model
from train import MovieRecTrainer
from utils import set_seed
from mlflow_tracking import MLFlowTracker


class MovieRecTuner:
    """
    Tuner for movie recommendation models.
    """
    
    def __init__(self, config: Dict[str, Any], mlflow_tracker: Optional[MLFlowTracker] = None):
        """
        Initialize MovieRecTuner.
        
        Args:
            config: Configuration dictionary
            mlflow_tracker: MLFlow tracker instance for experiment tracking
        """
        self.config = config
        self.mlflow_tracker = mlflow_tracker
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set random seed for reproducibility
        set_seed(config.get("seed", 42))
        
        logging.info(f"Hyperparameter tuning will run on: {self.device}")
    
    def train_function(self, config: Dict[str, Any], 
                      data_config: Dict[str, Any], 
                      checkpoint_dir: Optional[str] = None) -> None:
        """
        Training function for Ray Tune.
        
        Args:
            config: Hyperparameter configuration from Ray Tune
            data_config: Data configuration
            checkpoint_dir: Checkpoint directory
        """
        # Create trainer with combined config
        combined_config = {
            **data_config,
            "model": {
                "num_users": data_config["num_users"],
                "num_items": data_config["num_items"],
                **config
            },
            "batch_size": config.get("batch_size", 256),
            "learning_rate": config.get("learning_rate", 0.001),
            "weight_decay": config.get("weight_decay", 0.0),
            "num_epochs": 1  # Train for one epoch at a time for Ray Tune
        }
        
        trainer = MovieRecTrainer(combined_config)
        
        # Resume from checkpoint if available
        start_epoch = 0
        if checkpoint_dir:
            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
            checkpoint = torch.load(checkpoint_path)
            trainer.model.load_state_dict(checkpoint["model_state_dict"])
            start_epoch = checkpoint["epoch"]
        
        # Load data
        data, data_info = trainer.load_data()
        dataloaders = trainer.create_dataloaders(data)
        
        # Initialize model
        model_type = data_config.get("model_type", "mf")
        model = get_model(model_type, combined_config["model"])
        model.to(self.device)
        
        # Set up optimizer and loss function
        optimizer = optim.Adam(
            model.parameters(), 
            lr=config["learning_rate"],
            weight_decay=config.get("weight_decay", 0.0)
        )
        criterion = nn.MSELoss()
        
        # Train for one epoch
        train_metrics = trainer.train_epoch(model, dataloaders["train"], optimizer, criterion)
        
        # Evaluate
        val_metrics = trainer.evaluate(model, dataloaders["val"], criterion)
        
        # Save checkpoint
        checkpoint_path = os.path.join(tune.get_trial_dir(), "checkpoint.pt")
        torch.save({
            "epoch": start_epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_metrics["loss"],
            "val_rmse": val_metrics["rmse"]
        }, checkpoint_path)
        
        # Report metrics to Ray Tune
        session.report({
            "train_loss": train_metrics["loss"],
            "train_rmse": train_metrics["rmse"],
            "val_loss": val_metrics["loss"],
            "val_rmse": val_metrics["rmse"],
            "val_mae": val_metrics["mae"]
        }, checkpoint=checkpoint_path)
    
    def get_search_space(self) -> Dict[str, Any]:
        """
        Get the search space for hyperparameter tuning.
        
        Returns:
            Search space dictionary
        """
        model_type = self.config.get("model_type", "mf")
        
        if model_type == "mf":
            return {
                "embedding_dim": tune.choice([32, 64, 100, 128, 256]),
                "learning_rate": tune.loguniform(1e-4, 1e-2),
                "weight_decay": tune.loguniform(1e-6, 1e-3),
                "batch_size": tune.choice([64, 128, 256, 512])
            }
        elif model_type == "ncf":
            return {
                "mf_embedding_dim": tune.choice([16, 32, 64]),
                "mlp_embedding_dim": tune.choice([16, 32, 64]),
                "mlp_layers": tune.choice([
                    [128, 64, 32],
                    [256, 128, 64],
                    [128, 64],
                    [256, 128]
                ]),
                "dropout_rate": tune.uniform(0.1, 0.5),
                "learning_rate": tune.loguniform(1e-4, 1e-2),
                "weight_decay": tune.loguniform(1e-6, 1e-3),
                "batch_size": tune.choice([64, 128, 256, 512])
            }
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def tune(self) -> Dict[str, Any]:
        """
        Run hyperparameter tuning.
        
        Returns:
            Dictionary containing best trial results and model
        """
        # Load data to get data_info
        trainer = MovieRecTrainer(self.config)
        data, data_info = trainer.load_data()
        
        # Create data config
        data_config = {
            "data_path": self.config.get("data_path", "data/movielens/ml-latest-small"),
            "num_users": data_info["num_users"],
            "num_items": data_info["num_items"],
            "model_type": self.config.get("model_type", "mf")
        }
        
        # Initialize Ray
        if not ray.is_initialized():
            ray.init(num_cpus=self.config.get("num_cpus", 4), num_gpus=self.config.get("num_gpus", 0))
        
        # Set up search space
        search_space = self.get_search_space()
        
        # Set up ASHA scheduler
        scheduler = ASHAScheduler(
            max_t=self.config.get("max_epochs", 10),
            grace_period=self.config.get("grace_period", 2),
            reduction_factor=self.config.get("reduction_factor", 2)
        )
        
        # Start MLFlow run
        if self.mlflow_tracker is not None:
            self.mlflow_tracker.start_run(run_name=f"{data_config['model_type']}-tuning")
            self.mlflow_tracker.log_params({
                "model_type": data_config["model_type"],
                "num_users": data_info["num_users"],
                "num_items": data_info["num_items"],
                "num_trials": self.config.get("num_trials", 10),
                "max_epochs": self.config.get("max_epochs", 10)
            })
        
        # Run tuning
        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(self.train_function, data_config=data_config),
                resources={"cpu": 1, "gpu": 0.5 if torch.cuda.is_available() else 0}
            ),
            tune_config=tune.TuneConfig(
                metric="val_rmse",
                mode="min",
                scheduler=scheduler,
                num_samples=self.config.get("num_trials", 10)
            ),
            param_space=search_space,
            run_config=ray.air.RunConfig(
                name=f"{data_config['model_type']}_tuning",
                local_dir=self.config.get("output_dir", "ray_results"),
                stop={"training_iteration": self.config.get("max_epochs", 10)}
            )
        )
        
        # Execute tuning
        logging.info("Starting hyperparameter tuning...")
        start_time = time.time()
        results = tuner.fit()
        total_time = time.time() - start_time
        
        # Get best trial
        best_trial = results.get_best_result(metric="val_rmse", mode="min")
        best_config = best_trial.config
        best_val_rmse = best_trial.metrics["val_rmse"]
        best_checkpoint = best_trial.checkpoint
        
        logging.info(f"Hyperparameter tuning completed in {total_time:.2f} seconds")
        logging.info(f"Best trial config: {best_config}")
        logging.info(f"Best val RMSE: {best_val_rmse:.4f}")
        
        # Train final model with best hyperparameters
        final_config = {
            **self.config,
            "model": {
                "num_users": data_info["num_users"],
                "num_items": data_info["num_items"],
                **{k: v for k, v in best_config.items() if k not in ["learning_rate", "weight_decay", "batch_size"]}
            },
            "learning_rate": best_config["learning_rate"],
            "weight_decay": best_config.get("weight_decay", 0.0),
            "batch_size": best_config["batch_size"]
        }
        
        # Log best hyperparameters to MLFlow
        if self.mlflow_tracker is not None:
            self.mlflow_tracker.log_params({
                **{f"best_{k}": v for k, v in best_config.items()},
                "best_val_rmse": best_val_rmse,
                "tuning_time": total_time
            })
            self.mlflow_tracker.end_run()
        
        # Train final model with best hyperparameters
        logging.info("Training final model with best hyperparameters...")
        final_trainer = MovieRecTrainer(final_config, self.mlflow_tracker)
        final_results = final_trainer.train()
        
        return {
            "best_config": best_config,
            "best_val_rmse": best_val_rmse,
            "final_model": final_results["model"],
            "final_model_path": final_results["model_path"],
            "final_test_metrics": final_results["test_metrics"],
            "data_info": data_info
        }