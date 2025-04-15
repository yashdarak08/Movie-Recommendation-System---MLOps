import pandas as pd
import os
import logging
import numpy as np
import random
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, List, Tuple, Union, Any


def setup_logger(name: str, level=logging.INFO) -> logging.Logger:
    """
    Set up a logger with the specified name and level.
    
    Args:
        name: Name of the logger
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Only set up handler if not already configured
    if not logger.handlers:
        logger.setLevel(level)
        
        # Create console handler
        handler = logging.StreamHandler()
        handler.setLevel(level)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(handler)
        
        # Prevent propagation to root logger
        logger.propagate = False
    
    return logger

# Create logger for this module
logger = setup_logger(__name__)


def setup_logging(level_name: str = "INFO") -> None:
    """
    Set up root logger configuration.
    
    Args:
        level_name: Logging level name (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    level = getattr(logging, level_name)
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_metrics(predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Calculate evaluation metrics for recommendation model.
    
    Args:
        predictions: Predicted ratings
        labels: Ground truth ratings
        
    Returns:
        Dictionary of metrics (RMSE, MAE)
    """
    # Calculate root mean squared error
    rmse = np.sqrt(mean_squared_error(labels, predictions))
    
    # Calculate mean absolute error
    mae = mean_absolute_error(labels, predictions)
    
    return {
        "rmse": rmse,
        "mae": mae
    }


def load_movielens_data(data_path: str, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and preprocess MovieLens dataset.
    
    Args:
        data_path: Path to the MovieLens dataset directory
        train_ratio: Ratio of data used for training vs testing
        
    Returns:
        Tuple of (full_data, train_data, test_data)
    """
    try:
        # Load ratings data
        ratings_file = os.path.join(data_path, "ratings.csv")
        
        if not os.path.exists(ratings_file):
            raise FileNotFoundError(f"Ratings file not found at {ratings_file}")
        
        logger.info(f"Loading ratings from {ratings_file}")
        ratings_df = pd.read_csv(ratings_file)
        
        # Basic data cleaning
        ratings_df = ratings_df.dropna()
        
        # Display data information
        logger.info(f"Loaded {len(ratings_df)} ratings from {ratings_df['userId'].nunique()} users on {ratings_df['movieId'].nunique()} movies")
        
        # Optional: Filter out users/items with few interactions for better model performance
        # Apply minimum threshold for number of ratings
        min_user_ratings = 5
        min_movie_ratings = 5
        
        user_counts = ratings_df['userId'].value_counts()
        movie_counts = ratings_df['movieId'].value_counts()
        
        valid_users = user_counts[user_counts >= min_user_ratings].index
        valid_movies = movie_counts[movie_counts >= min_movie_ratings].index
        
        filtered_df = ratings_df[
            ratings_df['userId'].isin(valid_users) & 
            ratings_df['movieId'].isin(valid_movies)
        ]
        
        logger.info(f"After filtering: {len(filtered_df)} ratings from {filtered_df['userId'].nunique()} users on {filtered_df['movieId'].nunique()} movies")
        
        # Split data into train and test sets
        train_df, test_df = train_test_split(
            filtered_df, 
            test_size=(1-train_ratio),
            stratify=filtered_df['userId'].values,  # Stratify by user to ensure all users appear in train set
            random_state=42
        )
        
        logger.info(f"Training set: {len(train_df)} samples")
        logger.info(f"Test set: {len(test_df)} samples")
        
        # Optional: Load movie metadata for additional features
        try:
            movies_file = os.path.join(data_path, "movies.csv")
            movies_df = pd.read_csv(movies_file)
            logger.info(f"Loaded information for {len(movies_df)} movies")
            
            # Example: Extract movie year from title
            movies_df['year'] = movies_df['title'].str.extract(r'\((\d{4})\)$')
            
            # Note: We're not using the movie metadata in the basic recommender,
            # but it would be useful for content-based filtering components
            
        except Exception as e:
            logger.warning(f"Could not load movie metadata: {e}")
        
        return filtered_df, train_df, test_df
        
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        # Return empty DataFrames as fallback
        empty_df = pd.DataFrame(columns=['userId', 'movieId', 'rating', 'timestamp'])
        return empty_df, empty_df, empty_df


def save_model_metadata(metadata: Dict[str, Any], path: str) -> None:
    """
    Save model metadata to file.
    
    Args:
        metadata: Dictionary of metadata
        path: Path to save file
    """
    import json
    
    try:
        with open(path, 'w') as f:
            json.dump(metadata, f)
        logger.info(f"Saved model metadata to {path}")
    except Exception as e:
        logger.error(f"Error saving model metadata: {e}")


def load_model_metadata(path: str) -> Dict[str, Any]:
    """
    Load model metadata from file.
    
    Args:
        path: Path to metadata file
        
    Returns:
        Dictionary of metadata
    """
    import json
    
    try:
        with open(path, 'r') as f:
            metadata = json.load(f)
        return metadata
    except Exception as e:
        logger.error(f"Error loading model metadata: {e}")
        return {}


def normalize_ratings(ratings: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Normalize ratings to zero mean and unit variance.
    
    Args:
        ratings: Array of ratings
        
    Returns:
        Tuple of (normalized_ratings, mean, std)
    """
    mean = np.mean(ratings)
    std = np.std(ratings)
    
    if std == 0:
        # Avoid division by zero
        return ratings - mean, mean, 1.0
    
    normalized_ratings = (ratings - mean) / std
    return normalized_ratings, mean, std


def denormalize_ratings(normalized_ratings: np.ndarray, mean: float, std: float) -> np.ndarray:
    """
    Convert normalized ratings back to original scale.
    
    Args:
        normalized_ratings: Array of normalized ratings
        mean: Mean value used for normalization
        std: Standard deviation used for normalization
        
    Returns:
        Array of denormalized ratings
    """
    return normalized_ratings * std + mean


def create_sparse_matrix(ratings_df: pd.DataFrame) -> Tuple[np.ndarray, Dict, Dict]:
    """
    Create sparse user-item matrix from ratings DataFrame.
    
    Args:
        ratings_df: DataFrame with columns 'userId', 'movieId', 'rating'
        
    Returns:
        Tuple of (sparse_matrix, user_mapping, item_mapping)
    """
    from scipy.sparse import csr_matrix
    
    # Create mappings for users and items
    user_mapping = {id: i for i, id in enumerate(ratings_df['userId'].unique())}
    item_mapping = {id: i for i, id in enumerate(ratings_df['movieId'].unique())}
    
    # Map user and item IDs to indices
    row = ratings_df['userId'].map(user_mapping)
    col = ratings_df['movieId'].map(item_mapping)
    data = ratings_df['rating'].values
    
    # Create sparse matrix
    shape = (len(user_mapping), len(item_mapping))
    sparse_matrix = csr_matrix((data, (row, col)), shape=shape)
    
    return sparse_matrix, user_mapping, item_mapping