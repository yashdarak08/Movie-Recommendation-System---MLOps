import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import mlflow
import os
import numpy as np
import logging
from typing import Dict, Any, Tuple
import argparse

from mlflow_tracking import start_mlflow_run
from utils import load_movielens_data, setup_logger
from models import MatrixFactorizationModel, NeuralCollaborativeFiltering

logger = setup_logger(__name__)

class MovieLensDataset(Dataset):
    """Dataset class for MovieLens data."""
    
    def __init__(self, ratings_df, user_mapping=None, item_mapping=None, train=True):
        """
        Initialize MovieLens dataset.
        
        Args:
            ratings_df: DataFrame containing user-item interactions
            user_mapping: Dict mapping original user IDs to consecutive integers
            item_mapping: Dict mapping original item IDs to consecutive integers
            train: Boolean indicating if dataset is for training (vs evaluation)
        """
        self.train = train
        
        # Create mappings if not provided
        if user_mapping is None:
            self.user_mapping = {id: i for i, id in enumerate(ratings_df['userId'].unique())}
        else:
            self.user_mapping = user_mapping
            
        if item_mapping is None:
            self.item_mapping = {id: i for i, id in enumerate(ratings_df['movieId'].unique())}
        else:
            self.item_mapping = item_mapping
        
        self.num_users = len(self.user_mapping)
        self.num_items = len(self.item_mapping)
        
        # Convert user and item IDs to indices
        self.users = [self.user_mapping[x] for x in ratings_df['userId']]
        self.items = [self.item_mapping[x] for x in ratings_df['movieId']]
        self.ratings = ratings_df['rating'].values.astype(np.float32)
        
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        user = self.users[idx]
        item = self.items[idx]
        rating = self.ratings[idx]
        
        return {
            'user_id': torch.tensor(user, dtype=torch.long),
            'item_id': torch.tensor(item, dtype=torch.long),
            'rating': torch.tensor(rating, dtype=torch.float)
        }
    
    def get_metadata(self):
        """Returns metadata about the dataset dimensions."""
        return {
            'num_users': self.num_users,
            'num_items': self.num_items,
            'user_mapping': self.user_mapping,
            'item_mapping': self.item_mapping
        }

def train_epoch(model, dataloader, optimizer, criterion, device) -> float:
    """Train model for one epoch."""
    model.train()
    running_loss = 0.0
    
    for batch in dataloader:
        user_id = batch['user_id'].to(device)
        item_id = batch['item_id'].to(device)
        rating = batch['rating'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(user_id, item_id)
        loss = criterion(output.view(-1), rating)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * user_id.size(0)
    
    return running_loss / len(dataloader.dataset)

def evaluate(model, dataloader, criterion, device) -> Dict[str, float]:
    """Evaluate model on validation data."""
    model.eval()
    running_loss = 0.0
    mse = 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            user_id = batch['user_id'].to(device)
            item_id = batch['item_id'].to(device)
            rating = batch['rating'].to(device)
            
            # Forward pass
            output = model(user_id, item_id)
            loss = criterion(output.view(-1), rating)
            
            # Calculate metrics
            running_loss += loss.item() * user_id.size(0)
            mse += ((output.view(-1) - rating) ** 2).sum().item()
    
    val_loss = running_loss / len(dataloader.dataset)
    val_mse = mse / len(dataloader.dataset)
    val_rmse = np.sqrt(val_mse)
    
    return {
        'val_loss': val_loss,
        'val_mse': val_mse,
        'val_rmse': val_rmse
    }

def run_training(config: Dict[str, Any] = None):
    """Run model training with configuration."""
    
    if config is None:
        config = {
            'model_type': 'ncf',  # 'mf' for Matrix Factorization, 'ncf' for Neural CF
            'embedding_dim': 64,
            'hidden_dims': [256, 128, 64],
            'learning_rate': 0.001,
            'batch_size': 1024,
            'num_epochs': 20,
            'weight_decay': 1e-5,
            'use_early_stopping': True,
            'patience': 5,
            'train_test_split': 0.8,
            'random_seed': 42,
            'data_path': os.path.join("..", "data", "movielens", "ml-latest-small")
        }
    
    # Set random seeds for reproducibility
    torch.manual_seed(config['random_seed'])
    np.random.seed(config['random_seed'])
    
    # Load and preprocess data
    logger.info("Loading MovieLens data...")
    data, train_data, test_data = load_movielens_data(
        data_path=config['data_path'], 
        train_ratio=config['train_test_split']
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
    
    logger.info(f"Dataset contains {num_users} users and {num_items} items")
    logger.info(f"Training set: {len(train_dataset)} samples")
    logger.info(f"Validation set: {len(val_dataset)} samples")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    # Initialize model based on configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs!")
    
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
    
    # Move model to device and use DataParallel if multiple GPUs
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    
    # Initialize loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Start MLFlow run
    with start_mlflow_run("training_experiment") as run:
        # Log parameters
        mlflow.log_params(config)
        mlflow.log_params({
            "num_users": num_users,
            "num_items": num_items,
            "model_type": config['model_type']
        })
        
        # Training loop
        best_val_rmse = float('inf')
        patience_counter = 0
        
        for epoch in range(config['num_epochs']):
            # Train
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            
            # Evaluate
            val_metrics = evaluate(model, val_loader, criterion, device)
            val_loss = val_metrics['val_loss']
            val_rmse = val_metrics['val_rmse']
            
            # Log metrics
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_rmse": val_rmse
            }, step=epoch)
            
            logger.info(f"Epoch {epoch+1}/{config['num_epochs']} - "
                        f"Train Loss: {train_loss:.4f}, "
                        f"Val Loss: {val_loss:.4f}, "
                        f"Val RMSE: {val_rmse:.4f}")
            
            # Check for improvement
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                patience_counter = 0
                
                # Save best model
                if isinstance(model, nn.DataParallel):
                    state_dict = model.module.state_dict()
                else:
                    state_dict = model.state_dict()
                
                checkpoint = {
                    'model_state_dict': state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': config,
                    'metadata': metadata,
                    'epoch': epoch,
                    'val_rmse': val_rmse
                }
                
                torch.save(checkpoint, "best_model.pt")
                logger.info(f"Saved new best model with Val RMSE: {val_rmse:.4f}")
                
                # Log the best model to MLflow
                mlflow.pytorch.log_model(model, "model")
                mlflow.log_artifact("best_model.pt", artifact_path="checkpoints")
            else:
                patience_counter += 1
            
            # Early stopping
            if config['use_early_stopping'] and patience_counter >= config['patience']:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Log final best metrics
        mlflow.log_metrics({
            "best_val_rmse": best_val_rmse
        })
        
        logger.info(f"Training completed. Best validation RMSE: {best_val_rmse:.4f}")
        
        # Register model in MLflow model registry
        try:
            mv = mlflow.register_model(f"runs:/{run.info.run_id}/model", "MovieRecommender")
            logger.info(f"Model registered as: {mv.name} version {mv.version}")
        except Exception as e:
            logger.warning(f"Could not register model: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train movie recommendation model")
    parser.add_argument("--config", type=str, help="Path to config YAML file")
    args = parser.parse_args()
    
    if args.config:
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        run_training(config)
    else:
        run_training()