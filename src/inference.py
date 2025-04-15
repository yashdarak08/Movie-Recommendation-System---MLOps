"""
Inference module for movie recommendation models.
"""

import os
import json
import time
import logging
from typing import Dict, List, Tuple, Union, Any, Optional

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Query, Depends
from pydantic import BaseModel
import uvicorn
from prometheus_client import Counter, Histogram, start_http_server

import mlflow.pytorch
from models import get_model
from utils import setup_logger, load_model_metadata, set_seed


# Set up logger
logger = setup_logger(__name__)

# Define metrics
PREDICTION_COUNT = Counter('movie_rec_predictions_total', 'Total number of predictions')
PREDICTION_LATENCY = Histogram('movie_rec_prediction_latency_seconds', 'Prediction latency in seconds')
RECOMMENDATION_COUNT = Counter('movie_rec_recommendations_total', 'Total number of recommendation requests')
RECOMMENDATION_LATENCY = Histogram('movie_rec_recommendation_latency_seconds', 'Recommendation latency in seconds')


# Define request/response models
class PredictionRequest(BaseModel):
    """Request model for single prediction."""
    user_id: int
    item_id: int


class PredictionResponse(BaseModel):
    """Response model for single prediction."""
    user_id: int
    item_id: int
    predicted_rating: float


class BatchPredictionRequest(BaseModel):
    """Request model for batch prediction."""
    user_ids: List[int]
    item_ids: List[int]


class BatchPredictionResponse(BaseModel):
    """Response model for batch prediction."""
    predictions: List[Dict[str, Any]]
    latency_ms: float


class RecommendationRequest(BaseModel):
    """Request model for recommendations."""
    user_id: int
    top_k: int = 10
    exclude_rated: bool = True


class RecommendationResponse(BaseModel):
    """Response model for recommendations."""
    user_id: int
    recommendations: List[Dict[str, Any]]
    latency_ms: float


class MovieRecommenderInference:
    """
    Inference class for movie recommendation models.
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None, 
                 model_uri: Optional[str] = None, 
                 metadata_path: Optional[str] = None,
                 device: Optional[str] = None):
        """
        Initialize MovieRecommenderInference.
        
        Args:
            model_path: Path to model checkpoint file
            model_uri: MLflow model URI
            metadata_path: Path to model metadata file
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.model = None
        self.metadata = None
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.movies_df = None
        
        logger.info(f"Initializing inference engine on device: {self.device}")
        
        # Load model
        if model_uri:
            self._load_model_from_mlflow(model_uri)
        elif model_path:
            self._load_model_from_checkpoint(model_path)
        else:
            raise ValueError("Either model_path or model_uri must be provided")
        
        # Load metadata if path provided
        if metadata_path and os.path.exists(metadata_path):
            self.metadata = load_model_metadata(metadata_path)
        
        # Load movies data if available
        self._load_movies_data()
    
    def _load_model_from_mlflow(self, model_uri: str) -> None:
        """
        Load model from MLflow.
        
        Args:
            model_uri: MLflow model URI
        """
        try:
            logger.info(f"Loading model from MLflow: {model_uri}")
            self.model = mlflow.pytorch.load_model(model_uri)
            self.model.eval()
            self.model.to(self.device)
            
            # Try to load metadata from MLflow
            try:
                client = mlflow.tracking.MlflowClient()
                
                # Parse model URI to get run ID
                if "models:" in model_uri:
                    parts = model_uri.split('/')
                    model_name = parts[-2]
                    model_version = parts[-1]
                    
                    # Get run_id from which model was created
                    model_version_info = client.get_model_version(model_name, model_version)
                    run_id = model_version_info.run_id
                else:
                    # Assuming format like "runs:/<run_id>/artifacts/model"
                    parts = model_uri.split('/')
                    run_id = parts[1]
                
                # Download metadata from artifacts
                artifact_path = client.download_artifacts(run_id, "metadata.json", ".")
                self.metadata = load_model_metadata(artifact_path)
                logger.info("Loaded model metadata from MLflow")
            except Exception as e:
                logger.warning(f"Could not load metadata from MLflow: {e}")
                self.metadata = {}
                
        except Exception as e:
            logger.error(f"Error loading model from MLflow: {e}")
            raise
    
    def _load_model_from_checkpoint(self, model_path: str) -> None:
        """
        Load model from checkpoint file.
        
        Args:
            model_path: Path to model checkpoint
        """
        try:
            logger.info(f"Loading model from checkpoint: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Get model metadata
            self.metadata = checkpoint.get('metadata', {})
            if not self.metadata:
                self.metadata = checkpoint.get('data_info', {})
            
            # Get model configuration
            config = checkpoint.get('model_config', {})
            if not config:
                config = checkpoint.get('config', {})
            
            # Get model type
            model_type = checkpoint.get('model_type', 'ncf')
            
            # Get model dimensions
            num_users = self.metadata.get('num_users', 0)
            num_items = self.metadata.get('num_items', 0)
            
            if num_users == 0 or num_items == 0:
                raise ValueError("Invalid metadata: num_users or num_items is 0")
            
            # Initialize model
            model_config = {
                "num_users": num_users,
                "num_items": num_items,
                **config
            }
            
            self.model = get_model(model_type, model_config)
            
            # Load state dict
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            self.model.to(self.device)
            
            logger.info(f"Successfully loaded {model_type} model")
            
        except Exception as e:
            logger.error(f"Error loading model from checkpoint: {e}")
            raise
    
    def _load_movies_data(self) -> None:
        """Load movies data if available."""
        try:
            # Try to find movies.csv in common locations
            possible_paths = [
                "data/movielens/ml-latest-small/movies.csv",
                "../data/movielens/ml-latest-small/movies.csv",
                "data/movies.csv",
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/movielens/ml-latest-small/movies.csv")
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    self.movies_df = pd.read_csv(path)
                    logger.info(f"Loaded {len(self.movies_df)} movies from {path}")
                    break
            
            if self.movies_df is None:
                logger.warning("Could not find movies data file")
                
        except Exception as e:
            logger.warning(f"Error loading movies data: {e}")
    
    def predict(self, user_id: int, item_id: int) -> float:
        """
        Predict rating for a user-item pair.
        
        Args:
            user_id: User ID
            item_id: Item ID
            
        Returns:
            Predicted rating
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Get internal indices from metadata
        user_mapping = self.metadata.get('user_mapping', {})
        item_mapping = self.metadata.get('item_mapping', {})
        
        # Convert mapping keys to string for JSON compatibility
        user_mapping = {str(k): v for k, v in user_mapping.items()} if user_mapping else {}
        item_mapping = {str(k): v for k, v in item_mapping.items()} if item_mapping else {}
        
        # Check if user/item IDs are in training data
        if str(user_id) not in user_mapping or str(item_id) not in item_mapping:
            logger.warning(f"User ID {user_id} or Item ID {item_id} not in training data")
            return 0.0
        
        # Get internal indices
        user_idx = user_mapping[str(user_id)]
        item_idx = item_mapping[str(item_id)]
        
        # Convert to tensors
        user_tensor = torch.tensor([user_idx], dtype=torch.long).to(self.device)
        item_tensor = torch.tensor([item_idx], dtype=torch.long).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            start_time = time.time()
            prediction = self.model(user_tensor, item_tensor)
            latency = time.time() - start_time
            
            # Update metrics
            PREDICTION_COUNT.inc()
            PREDICTION_LATENCY.observe(latency)
        
        # Get prediction value
        if isinstance(prediction, torch.Tensor):
            if prediction.dim() > 1:
                prediction = prediction.squeeze()
            prediction_value = prediction.item()
            
        logger.debug(f"Prediction latency: {latency*1000:.2f} ms")
        return prediction_value
    
    def batch_predict(self, user_ids: List[int], item_ids: List[int]) -> Tuple[List[float], float]:
        """
        Predict ratings for multiple user-item pairs.
        
        Args:
            user_ids: List of user IDs
            item_ids: List of item IDs
            
        Returns:
            Tuple of (predictions, latency_ms)
        """
        if len(user_ids) != len(item_ids):
            raise ValueError("Length of user_ids and item_ids must match")
        
        if len(user_ids) == 0:
            return [], 0.0
        
        # Get internal indices from metadata
        user_mapping = self.metadata.get('user_mapping', {})
        item_mapping = self.metadata.get('item_mapping', {})
        
        # Convert mapping keys to string for JSON compatibility
        user_mapping = {str(k): v for k, v in user_mapping.items()} if user_mapping else {}
        item_mapping = {str(k): v for k, v in item_mapping.items()} if item_mapping else {}
        
        # Create lists for valid pairs
        valid_pairs = []
        valid_indices = []
        
        for i, (user_id, item_id) in enumerate(zip(user_ids, item_ids)):
            if str(user_id) in user_mapping and str(item_id) in item_mapping:
                valid_pairs.append((user_mapping[str(user_id)], item_mapping[str(item_id)]))
                valid_indices.append(i)
            else:
                logger.warning(f"User ID {user_id} or Item ID {item_id} not in training data")
        
        if not valid_pairs:
            return [0.0] * len(user_ids), 0.0
        
        # Convert to tensors
        user_tensor = torch.tensor([p[0] for p in valid_pairs], dtype=torch.long).to(self.device)
        item_tensor = torch.tensor([p[1] for p in valid_pairs], dtype=torch.long).to(self.device)
        
        # Make predictions
        with torch.no_grad():
            start_time = time.time()
            predictions = self.model(user_tensor, item_tensor)
            latency = time.time() - start_time
            
            # Update metrics
            PREDICTION_COUNT.inc(len(valid_pairs))
            PREDICTION_LATENCY.observe(latency)
        
        # Process predictions
        if isinstance(predictions, torch.Tensor):
            if predictions.dim() > 1:
                predictions = predictions.squeeze()
            pred_values = predictions.cpu().numpy().tolist()
        else:
            pred_values = predictions
            
        # Map predictions back to original order
        all_predictions = [0.0] * len(user_ids)
        for i, pred in zip(valid_indices, pred_values):
            all_predictions[i] = float(pred)
        
        logger.debug(f"Batch prediction latency: {latency*1000:.2f} ms for {len(valid_pairs)} items")
        return all_predictions, latency * 1000  # Return latency in ms
    
    def recommend(self, user_id: int, top_k: int = 10, exclude_rated: bool = True) -> Tuple[List[Dict[str, Any]], float]:
        """
        Generate recommendations for a user.
        
        Args:
            user_id: User ID
            top_k: Number of recommendations to return
            exclude_rated: Whether to exclude items the user has already rated
            
        Returns:
            Tuple of (recommendations, latency_ms)
        """
        start_time = time.time()
        
        # Get user's rated items if needed
        rated_items = set()
        if exclude_rated:
            # Try to get from metadata
            user_to_items = self.metadata.get('user_to_items', {})
            if user_to_items and str(user_id) in user_to_items:
                rated_items = set(user_to_items[str(user_id)])
        
        # Get all possible items
        user_mapping = self.metadata.get('user_mapping', {})
        item_mapping = self.metadata.get('item_mapping', {})
        
        # Convert mapping keys to string for JSON compatibility
        user_mapping = {str(k): v for k, v in user_mapping.items()} if user_mapping else {}
        item_mapping = {str(k): v for k, v in item_mapping.items()} if item_mapping else {}
        
        # Create reverse mapping (index -> item_id)
        reverse_item_mapping = {v: int(k) for k, v in item_mapping.items()}
        
        if str(user_id) not in user_mapping:
            logger.warning(f"User ID {user_id} not in training data")
            return [], 0.0
        
        # Get all candidate items
        all_item_ids = list(reverse_item_mapping.values())
        candidate_items = [item_id for item_id in all_item_ids if item_id not in rated_items]
        
        # Get predictions for all candidate items
        predictions, _ = self.batch_predict([user_id] * len(candidate_items), candidate_items)
        
        # Create item-prediction pairs
        item_preds = list(zip(candidate_items, predictions))
        
        # Sort by prediction score (descending)
        item_preds.sort(key=lambda x: x[1], reverse=True)
        
        # Get top-k items
        top_items = item_preds[:top_k]
        
        # Create recommendation list
        recommendations = []
        for item_id, score in top_items:
            item_info = {"item_id": item_id, "score": score}
            
            # Add movie information if available
            if self.movies_df is not None:
                movie_row = self.movies_df[self.movies_df['movieId'] == item_id]
                if not movie_row.empty:
                    movie_data = movie_row.iloc[0]
                    item_info["title"] = movie_data.get('title', '')
                    item_info["genres"] = movie_data.get('genres', '')
            
            recommendations.append(item_info)
        
        # Calculate total latency
        latency = time.time() - start_time
        
        # Update metrics
        RECOMMENDATION_COUNT.inc()
        RECOMMENDATION_LATENCY.observe(latency)
        
        logger.info(f"Recommendation latency: {latency*1000:.2f} ms for user {user_id}")
        return recommendations, latency * 1000  # Return latency in ms


# Create FastAPI app
app = FastAPI(title="Movie Recommendation API", version="1.0")
recommender = None


def get_recommender():
    """Get or initialize recommender instance."""
    global recommender
    if recommender is None:
        # Load configuration
        config_path = os.environ.get('CONFIG_PATH', 'configs/infer_config.yaml')
        if os.path.exists(config_path):
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = {}
        
        # Initialize recommender
        model_path = config.get('model', {}).get('path')
        model_uri = config.get('model', {}).get('uri')
        metadata_path = config.get('model', {}).get('metadata_path')
        
        if not model_path and not model_uri:
            raise ValueError("Either model path or model URI must be provided in config")
        
        recommender = MovieRecommenderInference(
            model_path=model_path,
            model_uri=model_uri,
            metadata_path=metadata_path
        )
    
    return recommender


# Define API endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check if recommender is initialized
        rec = get_recommender()
        return {"status": "healthy"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, recommender=Depends(get_recommender)):
    """
    Predict rating for a user-item pair.
    """
    try:
        prediction = recommender.predict(request.user_id, request.item_id)
        return {
            "user_id": request.user_id,
            "item_id": request.item_id,
            "predicted_rating": prediction
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictionRequest, recommender=Depends(get_recommender)):
    """
    Predict ratings for multiple user-item pairs.
    """
    try:
        predictions, latency = recommender.batch_predict(request.user_ids, request.item_ids)
        
        return {
            "predictions": [
                {"user_id": u, "item_id": i, "predicted_rating": p}
                for u, i, p in zip(request.user_ids, request.item_ids, predictions)
            ],
            "latency_ms": latency
        }
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommend", response_model=RecommendationResponse)
async def recommend(request: RecommendationRequest, recommender=Depends(get_recommender)):
    """
    Generate recommendations for a user.
    """
    try:
        recommendations, latency = recommender.recommend(
            request.user_id, 
            top_k=request.top_k,
            exclude_rated=request.exclude_rated
        )
        
        return {
            "user_id": request.user_id,
            "recommendations": recommendations,
            "latency_ms": latency
        }
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def run_inference_server(host: str = "0.0.0.0", port: int = 8000, metrics_port: int = 8001):
    """
    Run the inference server.
    
    Args:
        host: Server host
        port: Server port
        metrics_port: Prometheus metrics port
    """
    # Start Prometheus metrics server
    start_http_server(metrics_port)
    logger.info(f"Prometheus metrics server started on port {metrics_port}")
    
    # Start FastAPI server
    logger.info(f"Starting inference server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


def run_inference_benchmark(config: Dict[str, Any]):
    """
    Run inference benchmark.
    
    Args:
        config: Benchmark configuration
    """
    logger.info("Starting inference benchmark")
    
    # Get recommender
    model_path = config.get('model', {}).get('path')
    model_uri = config.get('model', {}).get('uri')
    metadata_path = config.get('model', {}).get('metadata_path')
    
    recommender = MovieRecommenderInference(
        model_path=model_path,
        model_uri=model_uri,
        metadata_path=metadata_path
    )
    
    # Benchmark parameters
    batch_sizes = config.get('benchmark', {}).get('batch_sizes', [1, 8, 32, 64, 128, 256])
    num_iterations = config.get('benchmark', {}).get('iterations', 10)
    warmup_iterations = config.get('benchmark', {}).get('warmup_iterations', 5)
    
    # Get metadata
    user_mapping = recommender.metadata.get('user_mapping', {})
    item_mapping = recommender.metadata.get('item_mapping', {})
    
    if not user_mapping or not item_mapping:
        logger.error("Missing metadata for benchmark")
        return
    
    # Convert to correct format
    user_mapping = {str(k): v for k, v in user_mapping.items()} if user_mapping else {}
    item_mapping = {str(k): v for k, v in item_mapping.items()} if item_mapping else {}
    
    # Get list of user and item IDs
    user_ids = [int(k) for k in user_mapping.keys()]
    item_ids = [int(k) for k in item_mapping.keys()]
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Results dictionary
    results = {}
    
    for batch_size in batch_sizes:
        logger.info(f"Benchmarking batch size: {batch_size}")
        
        # Warmup
        for _ in range(warmup_iterations):
            batch_user_ids = np.random.choice(user_ids, batch_size).tolist()
            batch_item_ids = np.random.choice(item_ids, batch_size).tolist()
            _ = recommender.batch_predict(batch_user_ids, batch_item_ids)
        
        # Benchmark
        latencies = []
        
        for _ in range(num_iterations):
            batch_user_ids = np.random.choice(user_ids, batch_size).tolist()
            batch_item_ids = np.random.choice(item_ids, batch_size).tolist()
            
            _, latency = recommender.batch_predict(batch_user_ids, batch_item_ids)
            latencies.append(latency)
        
        # Calculate statistics
        avg_latency = np.mean(latencies)
        p50_latency = np.percentile(latencies, 50)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        results[batch_size] = {
            'avg_latency_ms': float(avg_latency),
            'p50_latency_ms': float(p50_latency),
            'p95_latency_ms': float(p95_latency),
            'p99_latency_ms': float(p99_latency),
            'throughput': float(batch_size * 1000 / avg_latency)  # items/second
        }
        
        logger.info(f"Batch size {batch_size}: "
                   f"Avg latency = {avg_latency:.2f}ms, "
                   f"P99 latency = {p99_latency:.2f}ms, "
                   f"Throughput = {results[batch_size]['throughput']:.2f} items/sec")
    
    # Save results
    output_file = config.get('benchmark', {}).get('output_file', 'benchmark_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Benchmark results saved to {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Movie recommendation inference")
    parser.add_argument("--config", type=str, default="configs/infer_config.yaml", help="Path to config file")
    parser.add_argument("--mode", type=str, choices=["serve", "benchmark"], default="serve", help="Run mode")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--metrics-port", type=int, default=8001, help="Metrics server port")
    
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        logger.warning(f"Config file {args.config} not found. Using defaults.")
        config = {}
    
    # Set environment variable for config path
    os.environ['CONFIG_PATH'] = args.config
    
    # Run in specified mode
    if args.mode == "serve":
        host = config.get('server', {}).get('host', args.host)
        port = config.get('server', {}).get('port', args.port)
        metrics_port = config.get('monitoring', {}).get('metrics_port', args.metrics_port)
        
        run_inference_server(host, port, metrics_port)
    else:  # benchmark
        run_inference_benchmark(config)