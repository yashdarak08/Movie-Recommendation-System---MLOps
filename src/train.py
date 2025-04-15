"""
Training module for movie recommendation models.
"""

import os
import logging
import time
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from models import get_model
from utils import calculate_metrics, set_seed
from mlflow_tracking import MLFlowTracker


class MovieLensDataset(Dataset):
    """
    PyTorch Dataset for MovieLens data.
    """
    
    def __init__(self, 
                 user_ids: np.ndarray, 
                 movie_ids: np.ndarray, 
                 ratings: np.ndarray):
        """
        Initialize MovieLens Dataset.
        
        Args:
            user_ids: Array of user IDs
            movie_ids: Array of movie IDs
            ratings: Array of ratings
        """
        self.user_ids = torch.tensor(user_ids, dtype=torch.long)
        self.movie_ids = torch.tensor(movie_ids, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float)
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.ratings)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            User ID, movie ID, and rating
        """
        return self.user_ids[idx], self.movie_ids[idx], self.ratings[idx]


class MovieRecTrainer:
    """
    Trainer for movie recommendation models.
    """
    
    def __init__(self, config: Dict[str, Any], mlflow_tracker: Optional[MLFlowTracker] = None):
        """
        Initialize MovieRecTrainer.
        
        Args:
            config: Configuration dictionary
            mlflow_tracker: MLFlow tracker instance for experiment tracking
        """
        self.config = config
        self.mlflow_tracker = mlflow_tracker
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set random seed for reproducibility
        set_seed(config.get("seed", 42))
        
        logging.info(f"Training will run on: {self.device}")
    
    def load_data(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Load and preprocess MovieLens dataset.
        
        Returns:
            data: Dictionary containing preprocessed data arrays
            data_info: Dictionary containing dataset information
        """
        data_path = self.config.get("data_path", "data/movielens/ml-latest-small")
        ratings_file = os.path.join(data_path, "ratings.csv")
        movies_file = os.path.join(data_path, "movies.csv")
        
        # Load ratings and movies
        logging.info(f"Loading data from {data_path}")
        ratings_df = pd.read_csv(ratings_file)
        movies_df = pd.read_csv(movies_file)
        
        # Encode user and movie IDs
        user_encoder = LabelEncoder()
        movie_encoder = LabelEncoder()
        
        user_ids = user_encoder.fit_transform(ratings_df["userId"].values)
        movie_ids = movie_encoder.fit_transform(ratings_df["movieId"].values)
        ratings = ratings_df["rating"].values
        
        # Split data into train, validation, and test sets
        train_indices, test_indices = train_test_split(
            np.arange(len(ratings)),
            test_size=self.config.get("test_size", 0.2),
            random_state=self.config.get("seed", 42)
        )
        
        train_indices, val_indices = train_test_split(
            train_indices,
            test_size=self.config.get("val_size", 0.25),  # 20% of 80% = 20% validation
            random_state=self.config.get("seed", 42)
        )
        
        # Create data dictionary
        data = {
            "train_user_ids": user_ids[train_indices],
            "train_movie_ids": movie_ids[train_indices],
            "train_ratings": ratings[train_indices],
            
            "val_user_ids": user_ids[val_indices],
            "val_movie_ids": movie_ids[val_indices],
            "val_ratings": ratings[val_indices],
            
            "test_user_ids": user_ids[test_indices],
            "test_movie_ids": movie_ids[test_indices],
            "test_ratings": ratings[test_indices]
        }
        
        # Create data info dictionary
        data_info = {
            "num_users": len(user_encoder.classes_),
            "num_items": len(movie_encoder.classes_),
            "num_ratings": len(ratings),
            "user_encoder": user_encoder,
            "movie_encoder": movie_encoder,
            "movies_df": movies_df
        }
        
        logging.info(f"Loaded dataset with {data_info['num_users']} users, {data_info['num_items']} movies, and {data_info['num_ratings']} ratings")
        logging.info(f"Train: {len(data['train_ratings'])}, Val: {len(data['val_ratings'])}, Test: {len(data['test_ratings'])}")
        
        return data, data_info
    
    def create_dataloaders(self, data: Dict[str, np.ndarray]) -> Dict[str, DataLoader]:
        """
        Create PyTorch DataLoaders for training.
        
        Args:
            data: Dictionary containing data arrays
            
        Returns:
            Dictionary of DataLoaders
        """
        batch_size = self.config.get("batch_size", 256)
        num_workers = self.config.get("num_workers", 4)
        
        # Create datasets
        train_dataset = MovieLensDataset(
            data["train_user_ids"], 
            data["train_movie_ids"], 
            data["train_ratings"]
        )
        
        val_dataset = MovieLensDataset(
            data["val_user_ids"], 
            data["val_movie_ids"], 
            data["val_ratings"]
        )
        
        test_dataset = MovieLensDataset(
            data["test_user_ids"], 
            data["test_movie_ids"], 
            data["test_ratings"]
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        return {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader
        }
    
    def train_epoch(self, model: nn.Module, 
                   loader: DataLoader, 
                   optimizer: torch.optim.Optimizer, 
                   criterion: nn.Module) -> Dict[str, float]:
        """
        Train model for one epoch.
        
        Args:
            model: PyTorch model
            loader: DataLoader for training data
            optimizer: PyTorch optimizer
            criterion: Loss function
            
        Returns:
            Dictionary of metrics
        """
        model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        for user_ids, movie_ids, ratings in loader:
            # Move data to device
            user_ids = user_ids.to(self.device)
            movie_ids = movie_ids.to(self.device)
            ratings = ratings.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            predictions = model(user_ids, movie_ids)
            loss = criterion(predictions, ratings)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item() * len(ratings)
            all_preds.extend(predictions.detach().cpu().numpy())
            all_labels.extend(ratings.detach().cpu().numpy())
        
        # Calculate metrics
        metrics = calculate_metrics(np.array(all_preds), np.array(all_labels))
        metrics["loss"] = total_loss / len(loader.dataset)
        
        return metrics
    
    def evaluate(self, model: nn.Module, 
                loader: DataLoader, 
                criterion: nn.Module) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            model: PyTorch model
            loader: DataLoader for evaluation data
            criterion: Loss function
            
        Returns:
            Dictionary of metrics
        """
        model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for user_ids, movie_ids, ratings in loader:
                # Move data to device
                user_ids = user_ids.to(self.device)
                movie_ids = movie_ids.to(self.device)
                ratings = ratings.to(self.device)
                
                # Forward pass
                predictions = model(user_ids, movie_ids)
                loss = criterion(predictions, ratings)
                
                # Track metrics
                total_loss += loss.item() * len(ratings)
                all_preds.extend(predictions.detach().cpu().numpy())
                all_labels.extend(ratings.detach().cpu().numpy())
        
        # Calculate metrics
        metrics = calculate_metrics(np.array(all_preds), np.array(all_labels))
        metrics["loss"] = total_loss / len(loader.dataset)
        
        return metrics
    
    def train(self) -> Dict[str, Any]:
        """
        Train the model.
        
        Returns:
            Dictionary containing trained model and metrics
        """
        # Load data
        data, data_info = self.load_data()
        dataloaders = self.create_dataloaders(data)
        
        # Update model config with data info
        model_config = self.config.get("model", {})
        model_config["num_users"] = data_info["num_users"]
        model_config["num_items"] = data_info["num_items"]
        
        # Get model
        model_type = self.config.get("model_type", "mf")
        model = get_model(model_type, model_config)
        model.to(self.device)
        
        # Start MLFlow run
        if self.mlflow_tracker is not None:
            self.mlflow_tracker.start_run(run_name=f"{model_type}-training")
            self.mlflow_tracker.log_params({
                "model_type": model_type,
                "num_users": data_info["num_users"],
                "num_items": data_info["num_items"],
                "train_size": len(data["train_ratings"]),
                "val_size": len(data["val_ratings"]),
                "test_size": len(data["test_ratings"]),
                **model_config
            })
        
        # Set up optimizer and loss function
        lr = self.config.get("learning_rate", 0.001)
        weight_decay = self.config.get("weight_decay", 0.0)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        
        # Training loop
        num_epochs = self.config.get("num_epochs", 10)
        best_val_rmse = float("inf")
        best_model_state = None
        patience = self.config.get("patience", 5)
        patience_counter = 0
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch(model, dataloaders["train"], optimizer, criterion)
            
            # Validate
            val_metrics = self.evaluate(model, dataloaders["val"], criterion)
            
            epoch_time = time.time() - epoch_start_time
            
            # Log metrics
            logging.info(f"Epoch {epoch+1}/{num_epochs} - "
                         f"Train Loss: {train_metrics['loss']:.4f}, "
                         f"Train RMSE: {train_metrics['rmse']:.4f}, "
                         f"Val Loss: {val_metrics['loss']:.4f}, "
                         f"Val RMSE: {val_metrics['rmse']:.4f}, "
                         f"Time: {epoch_time:.2f}s")
            
            if self.mlflow_tracker is not None:
                self.mlflow_tracker.log_metrics({
                    "train_loss": train_metrics["loss"],
                    "train_rmse": train_metrics["rmse"],
                    "train_mae": train_metrics["mae"],
                    "val_loss": val_metrics["loss"],
                    "val_rmse": val_metrics["rmse"],
                    "val_mae": val_metrics["mae"],
                    "epoch_time": epoch_time
                }, step=epoch)
            
            # Check for improvement
            if val_metrics["rmse"] < best_val_rmse:
                best_val_rmse = val_metrics["rmse"]
                best_model_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                logging.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        # Test best model
        test_metrics = self.evaluate(model, dataloaders["test"], criterion)
        logging.info(f"Test RMSE: {test_metrics['rmse']:.4f}, Test MAE: {test_metrics['mae']:.4f}")
        
        if self.mlflow_tracker is not None:
            self.mlflow_tracker.log_metrics({
                "test_loss": test_metrics["loss"],
                "test_rmse": test_metrics["rmse"],
                "test_mae": test_metrics["mae"]
            })
        
        # Save model
        output_dir = self.config.get("output_dir", "models")
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, f"{model_type}_model.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
            "model_type": model_type,
            "data_info": {
                "num_users": data_info["num_users"],
                "num_items": data_info["num_items"]
            }
        }, model_path)
        
        logging.info(f"Model saved to {model_path}")
        
        if self.mlflow_tracker is not None:
            self.mlflow_tracker.log_artifact(model_path)
            self.mlflow_tracker.end_run()
        
        return {
            "model": model,
            "model_path": model_path,
            "data_info": data_info,
            "test_metrics": test_metrics
        }