import mlflow.pytorch
import torch
import time
import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Any, Optional
from flask import Flask, request, jsonify
import sys
import argparse

from utils import setup_logger, load_model_metadata

logger = setup_logger(__name__)

class RecommenderInference:
    """
    Inference class for recommendation models.
    """
    
    def __init__(self, model_path: str = None, model_uri: str = None, device: str = None):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to stored model checkpoint
            model_uri: MLflow model URI
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.model = None
        self.metadata = None
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Initializing inference engine on device: {self.device}")
        
        # Load model
        if model_uri:
            self._load_model_from_mlflow(model_uri)
        elif model_path:
            self._load_model_from_checkpoint(model_path)
        else:
            raise ValueError("Either model_path or model_uri must be provided")
    
    def _load_model_from_mlflow(self, model_uri: str) -> None:
        """
        Load model from MLflow.
        
        Args:
            model_uri: MLflow model URI (e.g., "models:/MovieRecommender/Production")
        """
        try:
            logger.info(f"Loading model from MLflow: {model_uri}")
            self.model = mlflow.pytorch.load_model(model_uri)
            self.model.eval()
            self.model.to(self.device)
            
            # Try to load metadata
            try:
                client = mlflow.tracking.MlflowClient()
                parts = model_uri.split('/')
                if "models:" in model_uri:
                    model_name = parts[-2]
                    model_version = parts[-1]
                    
                    # Get run_id from which model was created
                    try:
                        model_version_info = client.get_model_version(model_name, model_version)
                        run_id = model_version_info.run_id
                    except Exception as e:
                        logger.warning(f"Could not get model version info: {e}")
                        raise
                else:
                    # Assuming format like "runs:/<run_id>/artifacts/model"
                    run_id = parts[1]
                
                # Download metadata from artifacts
                try:
                    artifact_path = client.download_artifacts(run_id, "metadata.json", ".")
                    self.metadata = load_model_metadata(artifact_path)
                    logger.info("Loaded model metadata from MLflow")
                except Exception as e:
                    logger.warning(f"Could not download metadata artifact: {e}")
                    raise
            except Exception as e:
                logger.warning(f"Could not load metadata from MLflow: {e}")
                self.metadata = {
                    "user_mapping": {},
                    "item_mapping": {}
                }
        except Exception as e:
            logger.error(f"Error loading model from MLflow: {e}")
            raise
    
    def _load_model_from_checkpoint(self, model_path: str) -> None:
        """
        Load model from local checkpoint.
        
        Args:
            model_path: Path to model checkpoint
        """
        try:
            logger.info(f"Loading model from checkpoint: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Load metadata from checkpoint
            self.metadata = checkpoint.get('metadata', {})
            
            # Import the model class dynamically
            from models import MatrixFactorizationModel, NeuralCollaborativeFiltering
            
            # Determine model type and initialize
            config = checkpoint.get('config', {})
            model_type = config.get('model_type', 'ncf')
            
            num_users = self.metadata.get('num_users', 0)
            num_items = self.metadata.get('num_items', 0)
            
            if num_users == 0 or num_items == 0:
                raise ValueError("Invalid metadata: num_users or num_items is 0")
            
            if model_type == 'mf':
                self.model = MatrixFactorizationModel(
                    num_users=num_users,
                    num_items=num_items,
                    embedding_dim=config.get('embedding_dim', 64)
                )
            else:  # 'ncf'
                self.model = NeuralCollaborativeFiltering(
                    num_users=num_users,
                    num_items=num_items,
                    embedding_dim=config.get('embedding_dim', 64),
                    hidden_dims=config.get('hidden_dims', [256, 128, 64])
                )
            
            # Load state dict
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            self.model.to(self.device)
            
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model from checkpoint: {e}")
            raise
    
    def predict(self, user_id: int, item_id: int) -> float:
        """
        Predict rating for a single user-item pair.
        
        Args:
            user_id: Original user ID
            item_id: Original item ID
            
        Returns:
            Predicted rating
        """
        # Map original IDs to internal indices
        user_mapping = self.metadata.get('user_mapping', {})
        item_mapping = self.metadata.get('item_mapping', {})
        
        # Convert to string keys for JSON compatibility
        user_mapping = {int(k): v for k, v in user_mapping.items()} if user_mapping else {}
        item_mapping = {int(k): v for k, v in item_mapping.items()} if item_mapping else {}
        
        if str(user_id) not in user_mapping or str(item_id) not in item_mapping:
            logger.warning(f"User ID {user_id} or Item ID {item_id} not in training data")
            return 0.0
        
        user_idx = user_mapping.get(str(user_id))
        item_idx = item_mapping.get(str(item_id))
        
        # Convert to tensors
        user_tensor = torch.tensor([user_idx], dtype=torch.long).to(self.device)
        item_tensor = torch.tensor([item_idx], dtype=torch.long).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            start_time = time.time()
            prediction = self.model(user_tensor, item_tensor)
            latency = (time.time() - start_time) * 1000  # ms
            
        logger.debug(f"Inference latency: {latency:.2f} ms")
        
        return prediction.item()
    
    def batch_predict(self, user_ids: List[int], item_ids: List[int]) -> List[float]:
        """
        Predict ratings for a batch of user-item pairs.
        
        Args:
            user_ids: List of user IDs
            item_ids: List of item IDs
            
        Returns:
            List of predicted ratings
        """
        if len(user_ids) != len(item_ids):
            raise ValueError("Length of user_ids and item_ids must match")
        
        if len(user_ids) == 0:
            return []
        
        # Map original IDs to internal indices
        user_mapping = self.metadata.get('user_mapping', {})
        item_mapping = self.metadata.get('item_mapping', {})
        
        # Convert to string keys for JSON compatibility
        user_mapping = {int(k): v for k, v in user_mapping.items()} if user_mapping else {}
        item_mapping = {int(k): v for k, v in item_mapping.items()} if item_mapping else {}
        
        # Filter valid pairs
        valid_pairs = []
        valid_indices = []
        
        for i, (user_id, item_id) in enumerate(zip(user_ids, item_ids)):
            if str(user_id) in user_mapping and str(item_id) in item_mapping:
                valid_pairs.append((user_mapping[str(user_id)], item_mapping[str(item_id)]))
                valid_indices.append(i)
            else:
                logger.warning(f"User ID {user_id} or Item ID {item_id} not in training data")
        
        if not valid_pairs:
            return [0.0] * len(user_ids)
        
        # Convert to tensors
        user_tensor = torch.tensor([p[0] for p in valid_pairs], dtype=torch.long).to(self.device)
        item_tensor = torch.tensor([p[1] for p in valid_pairs], dtype=torch.long).to(self.device)
        
        # Make predictions
        with torch.no_grad():
            start_time = time.time()
            predictions = self.model(user_tensor, item_tensor)
            latency = (time.time() - start_time) * 1000  # ms
            
        logger.debug(f"Batch inference latency: {latency:.2f} ms for {len(valid_pairs)} items")
        
        # Map predictions back to original order with zeros for invalid pairs
        all_predictions = [0.0] * len(user_ids)
        for i, pred in zip(valid_indices, predictions.cpu().numpy()):
            all_predictions[i] = float(pred)
        
        return all_predictions
    
    def recommend_items(self, user_id: int, top_k: int = 10, exclude_rated: bool = True) -> List[Tuple[int, float]]:
        """
        Recommend top-k items for a user.
        
        Args:
            user_id: User ID
            top_k: Number of recommendations to generate
            exclude_rated: Whether to exclude items the user has already rated
            
        Returns:
            List of (item_id, score) tuples
        """
        user_mapping = self.metadata.get('user_mapping', {})
        item_mapping = self.metadata.get('item_mapping', {})
        
        # Convert to string keys for JSON compatibility
        user_mapping = {int(k): v for k, v in user_mapping.items()} if user_mapping else {}
        item_mapping = {int(k): v for k, v in item_mapping.items()} if item_mapping else {}
        
        # Create reverse mapping (index -> item_id)
        reverse_item_mapping = {v: int(k) for k, v in item_mapping.items()}
        
        if str(user_id) not in user_mapping:
            logger.warning(f"User ID {user_id} not in training data")
            return []
        
        user_idx = user_mapping[str(user_id)]
        
        # Get all items
        all_items = list(reverse_item_mapping.keys())
        
        # Batch prediction for all items
        user_tensor = torch.tensor([user_idx] * len(all_items), dtype=torch.long).to(self.device)
        item_tensor = torch.tensor(all_items, dtype=torch.long).to(self.device)
        
        # Make predictions
        with torch.no_grad():
            start_time = time.time()
            predictions = self.model(user_tensor, item_tensor)
            latency = (time.time() - start_time) * 1000  # ms
            
        logger.info(f"Recommendation latency: {latency:.2f} ms for {len(all_items)} items")
        
        # Convert to list of (item_id, score) tuples
        item_scores = [(reverse_item_mapping[idx], score.item()) 
                      for idx, score in zip(all_items, predictions)]
        
        # Sort by score (descending)
        item_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k
        return item_scores[:top_k]


# Flask API for serving recommendations
app = Flask(__name__)
inference_engine = None

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    return jsonify({"status": "healthy"})

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for predicting a single user-item rating."""
    data = request.json
    user_id = data.get('user_id')
    item_id = data.get('item_id')
    
    if user_id is None or item_id is None:
        return jsonify({"error": "Missing user_id or item_id"}), 400
    
    try:
        prediction = inference_engine.predict(user_id, item_id)
        
        # Record metrics (e.g., for Prometheus)
        # This would normally require a Prometheus client library
        
        return jsonify({
            "user_id": user_id,
            "item_id": item_id,
            "predicted_rating": prediction
        })
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Endpoint for batch predictions."""
    data = request.json
    user_ids = data.get('user_ids', [])
    item_ids = data.get('item_ids', [])
    
    if not user_ids or not item_ids or len(user_ids) != len(item_ids):
        return jsonify({"error": "Invalid input format"}), 400
    
    try:
        predictions = inference_engine.batch_predict(user_ids, item_ids)
        
        return jsonify({
            "predictions": [
                {"user_id": u, "item_id": i, "predicted_rating": p}
                for u, i, p in zip(user_ids, item_ids, predictions)
            ]
        })
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/recommend', methods=['POST'])
def recommend():
    """Endpoint for generating recommendations for a user."""
    data = request.json
    user_id = data.get('user_id')
    top_k = data.get('top_k', 10)
    
    if user_id is None:
        return jsonify({"error": "Missing user_id"}), 400
    
    try:
        recommendations = inference_engine.recommend_items(user_id, top_k)
        
        return jsonify({
            "user_id": user_id,
            "recommendations": [
                {"item_id": item_id, "score": score}
                for item_id, score in recommendations
            ]
        })
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        return jsonify({"error": str(e)}), 500


def run_inference(config: Dict[str, Any] = None):
    """
    Run inference server or standalone benchmark.
    
    Args:
        config: Configuration dictionary
    """
    global inference_engine
    
    if config is None:
        config = {
            'model_uri': "models:/MovieRecommender/Production",
            'model_path': None,
            'batch_size': 64,
            'mode': 'server',  # 'server' or 'benchmark'
            'host': '0.0.0.0',
            'port': 8000,
            'log_level': 'INFO'
        }
    
    # Set up logging
    log_level = getattr(logging, config.get('log_level', 'INFO'))
    logging.basicConfig(level=log_level)
    
    # Initialize inference engine
    model_uri = config.get('model_uri')
    model_path = config.get('model_path')
    
    inference_engine = RecommenderInference(
        model_path=model_path,
        model_uri=model_uri
    )
    
    mode = config.get('mode', 'server')
    
    if mode == 'server':
        # Run Flask server
        app.run(
            host=config.get('host', '0.0.0.0'),
            port=config.get('port', 8000)
        )
    elif mode == 'benchmark':
        # Run benchmark
        logger.info("Running inference benchmark...")
        _run_benchmark(inference_engine, config)
    else:
        logger.error(f"Invalid mode: {mode}")
        sys.exit(1)


def _run_benchmark(engine: RecommenderInference, config: Dict[str, Any]):
    """
    Run an inference benchmark.
    
    Args:
        engine: Initialized RecommenderInference instance
        config: Configuration dictionary
    """
    batch_sizes = config.get('benchmark_batch_sizes', [1, 8, 32, 64, 128, 256])
    num_iterations = config.get('benchmark_iterations', 100)
    
    results = {}
    
    for batch_size in batch_sizes:
        logger.info(f"Benchmarking batch size: {batch_size}")
        
        # Create random user-item pairs for benchmarking
        user_mapping = engine.metadata.get('user_mapping', {})
        item_mapping = engine.metadata.get('item_mapping', {})
        
        if not user_mapping or not item_mapping:
            logger.error("Missing metadata for benchmark")
            return
        
        # Get a list of valid user and item IDs
        user_ids = list(user_mapping.keys())
        item_ids = list(item_mapping.keys())
        
        # For reproducibility
        np.random.seed(42)
        
        latencies = []
        
        for i in range(num_iterations):
            # Generate random batch
            batch_user_ids = np.random.choice(user_ids, batch_size).tolist()
            batch_item_ids = np.random.choice(item_ids, batch_size).tolist()
            
            # Convert string keys to integers
            batch_user_ids = [int(u) for u in batch_user_ids]
            batch_item_ids = [int(i) for i in batch_item_ids]
            
            # Measure inference time
            start_time = time.time()
            _ = engine.batch_predict(batch_user_ids, batch_item_ids)
            latency = (time.time() - start_time) * 1000  # ms
            
            latencies.append(latency)
        
        # Calculate statistics
        avg_latency = np.mean(latencies)
        p50_latency = np.percentile(latencies, 50)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        results[batch_size] = {
            'avg_latency_ms': avg_latency,
            'p50_latency_ms': p50_latency,
            'p95_latency_ms': p95_latency,
            'p99_latency_ms': p99_latency,
            'throughput': batch_size * 1000 / avg_latency  # items/second
        }
        
        logger.info(f"Batch size {batch_size}: Avg latency = {avg_latency:.2f}ms, "
                   f"P99 latency = {p99_latency:.2f}ms, "
                   f"Throughput = {results[batch_size]['throughput']:.2f} items/sec")
    
    # Save results
    output_file = config.get('benchmark_output', 'benchmark_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Benchmark results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference server or benchmark")
    parser.add_argument("--config", type=str, help="Path to config YAML file")
    parser.add_argument("--mode", type=str, choices=["server", "benchmark"], 
                        default="server", help="Mode to run")
    parser.add_argument("--model_uri", type=str, help="MLflow model URI")
    parser.add_argument("--model_path", type=str, help="Path to model checkpoint")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    
    args = parser.parse_args()
    
    if args.config:
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {
            'mode': args.mode,
            'port': args.port
        }
        
        if args.model_uri:
            config['model_uri'] = args.model_uri
        elif args.model_path:
            config['model_path'] = args.model_path
        else:
            # Default to latest production model
            config['model_uri'] = "models:/MovieRecommender/Production"
    
    run_inference(config)