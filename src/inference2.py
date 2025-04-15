"""
Inference module for movie recommendation models.
"""

import os
import time
import logging
import pickle
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import uvicorn
from prometheus_client import Counter, Histogram, start_http_server

from models import get_model
from utils import set_seed


# Define metrics
PREDICTION_COUNT = Counter('movie_rec_predictions_total', 'Total number of predictions')
PREDICTION_LATENCY = Histogram('movie_rec_prediction_latency_seconds', 'Prediction latency in seconds')


class MoviePredictionRequest(BaseModel):
    """Request model for movie prediction."""
    user_id: int
    movie_ids: List[int]


class MoviePredictionResponse(BaseModel):
    """Response model for movie prediction."""
    predictions: Dict[str, float]
    latency_ms: float


class MovieRecommendationRequest(BaseModel):
    """Request model for movie recommendations."""
    user_id: int
    n: int = 10
    include_seen: bool = False


class MovieRecommendationResponse(BaseModel):
    """Response model for movie recommendations."""
    recommendations: List[Dict[str, Any]]
    latency_ms: float


class MovieRecInference:
    """
    Inference class for movie recommendation models.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MovieRecInference.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.data_info = None
        self.movies_df = None
        self.user_ratings = None
        
        # Set random seed for reproducibility
        set_seed(config.get("seed", 42))
        
        # Load model if path is provided
        model_path = config.get("model_path")
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        logging.info(f"Inference will run on: {self.device}")
    
    def load_model(self, model_path: str) -> None:
        """
        Load a trained model.
        
        Args:
            model_path: Path to the model file
        """
        logging.info(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        model_type = checkpoint["model_type"]
        model_config = checkpoint["model_config"]
        data_info = checkpoint["data_info"]
        
        self.model = get_model(model_type, model_config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        
        self.data_info = data_info
        
        logging.info(f"Loaded {model_type} model with {model_config['embedding_dim'] if 'embedding_dim' in model_config else model_config['mf_embedding_dim']} embedding dimensions")
        
        # Load movies data if available
        data_path = self.config.get("data_path", "data/movielens/ml-latest-small")
        movies_file = os.path.join(data_path, "movies.csv")
        if os.path.exists(movies_file):
            self.movies_df = pd.read_csv(movies_file)
            logging.info(f"Loaded {len(self.movies_df)} movies from {movies_file}")
        
        # Load user ratings for recommendation filtering if needed
        ratings_file = os.path.join(data_path, "ratings.csv")
        if os.path.exists(ratings_file):
            ratings_df = pd.read_csv(ratings_file)
            self.user_ratings = {
                user_id: set(movie_ids) 
                for user_id, movie_ids in ratings_df.groupby("userId")["movieId"].apply(list).items()
            }
            logging.info(f"Loaded ratings for {len(self.user_ratings)} users from {ratings_file}")
    
    def predict(self, user_id: int, movie_ids: List[int]) -> Dict[int, float]:
        """
        Predict ratings for a user and list of movies.
        
        Args:
            user_id: User ID
            movie_ids: List of movie IDs
            
        Returns:
            Dictionary of movie IDs and predicted ratings
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Convert IDs to indices
        try:
            # We need to handle the case where the user or movie ID might not be in the training set
            if not hasattr(self, "user_encoder") or not hasattr(self, "movie_encoder"):
                # Load data encoders
                data_path = self.config.get("data_path", "data/movielens/ml-latest-small")
                encoders_file = os.path.join(data_path, "encoders.pkl")
                if os.path.exists(encoders_file):
                    with open(encoders_file, "rb") as f:
                        encoders = pickle.load(f)
                        self.user_encoder = encoders["user_encoder"]
                        self.movie_encoder = encoders["movie_encoder"]
                else:
                    # If encoders are not available, use simple mapping
                    # This is a fallback and won't be accurate
                    user_id_idx = user_id % self.data_info["num_users"]
                    movie_id_indices = [movie_id % self.data_info["num_items"] for movie_id in movie_ids]
                    logging.warning("Encoders not found. Using fallback mapping which may not be accurate.")
            else:
                # Use proper encoding
                user_id_idx = self.user_encoder.transform([user_id])[0]
                movie_id_indices = self.movie_encoder.transform(movie_ids)
        except (ValueError, KeyError) as e:
            logging.warning(f"Error encoding IDs: {e}. Using fallback mapping.")
            # Fallback to simple mapping
            user_id_idx = user_id % self.data_info["num_users"]
            movie_id_indices = [movie_id % self.data_info["num_items"] for movie_id in movie_ids]
        
        # Create tensors
        user_indices = torch.LongTensor([user_id_idx] * len(movie_id_indices)).to(self.device)
        movie_indices = torch.LongTensor(movie_id_indices).to(self.device)
        
        # Make predictions
        with torch.no_grad():
            start_time = time.time()
            predictions = self.model(user_indices, movie_indices).cpu().numpy()
            latency = time.time() - start_time
        
        # Update metrics
        PREDICTION_COUNT.inc(len(movie_ids))
        PREDICTION_LATENCY.observe(latency)
        
        # Create prediction dictionary
        prediction_dict = {
            movie_id: float(pred) for movie_id, pred in zip(movie_ids, predictions)
        }
        
        return prediction_dict, latency * 1000  # Convert to ms
    
    def recommend(self, user_id: int, n: int = 10, include_seen: bool = False) -> Tuple[List[Dict[str, Any]], float]:
        """
        Recommend top N movies for a user.
        
        Args:
            user_id: User ID
            n: Number of recommendations
            include_seen: Whether to include movies the user has already seen
            
        Returns:
            List of recommended movies with predicted ratings
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        if self.movies_df is None:
            raise ValueError("Movies data not loaded.")
        
        # Get all movie IDs
        all_movie_ids = self.movies_df["movieId"].tolist()
        
        # Filter out seen movies if needed
        if not include_seen and self.user_ratings and user_id in self.user_ratings:
            seen_movies = self.user_ratings[user_id]
            candidate_movies = [movie_id for movie_id in all_movie_ids if movie_id not in seen_movies]
        else:
            candidate_movies = all_movie_ids
        
        # Get predictions for all candidate movies
        predictions, latency = self.predict(user_id, candidate_movies)
        
        # Sort by predicted rating
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        top_n = sorted_predictions[:n]
        
        # Create recommendation list with movie details
        recommendations = []
        for movie_id, rating in top_n:
            movie_info = self.movies_df[self.movies_df["movieId"] == movie_id].iloc[0].to_dict()
            recommendations.append({
                "movie_id": int(movie_id),
                "title": movie_info["title"],
                "genres": movie_info["genres"],
                "predicted_rating": float(rating)
            })
        
        return recommendations, latency
    
    def run_inference(self) -> None:
        """Run batch inference on test data."""
        data_path = self.config.get("data_path", "data/movielens/ml-latest-small")
        test_file = os.path.join(data_path, "test_ratings.csv")
        output_file = self.config.get("output_file", "predictions.csv")
        
        if not os.path.exists(test_file):
            logging.error(f"Test file not found: {test_file}")
            return
        
        # Load test data
        test_df = pd.read_csv(test_file)
        logging.info(f"Loaded {len(test_df)} test samples from {test_file}")
        
        # Group by user for batch prediction
        predictions = []
        start_time = time.time()
        
        for user_id, group in test_df.groupby("userId"):
            movie_ids = group["movieId"].tolist()
            user_predictions, _ = self.predict(user_id, movie_ids)
            
            for movie_id in movie_ids:
                predictions.append({
                    "userId": user_id,
                    "movieId": movie_id,
                    "predicted_rating": user_predictions.get(movie_id, 0.0)
                })
        
        total_time = time.time() - start_time
        
        # Convert to DataFrame and save
        predictions_df = pd.DataFrame(predictions)
        predictions_df.to_csv(output_file, index=False)
        
        logging.info(f"Inference completed in {total_time:.2f} seconds")
        logging.info(f"Predictions saved to {output_file}")
    
    def start_service(self) -> None:
        """Start the prediction service with FastAPI."""
        app = FastAPI(title="Movie Recommendation API", version="1.0.0")
        
        # Start Prometheus metrics server
        metrics_port = self.config.get("metrics_port", 8001)
        start_http_server(metrics_port)
        logging.info(f"Prometheus metrics server started on port {metrics_port}")
        
        @app.get("/")
        async def root():
            return {"message": "Movie Recommendation API"}
        
        @app.post("/predict", response_model=MoviePredictionResponse)
        async def predict_endpoint(request: MoviePredictionRequest):
            try:
                predictions, latency = self.predict(request.user_id, request.movie_ids)
                return MoviePredictionResponse(
                    predictions={str(k): v for k, v in predictions.items()},
                    latency_ms=latency
                )
            except Exception as e:
                logging.error(f"Prediction error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/recommend", response_model=MovieRecommendationResponse)
        async def recommend_endpoint(request: MovieRecommendationRequest):
            try:
                recommendations, latency = self.recommend(
                    request.user_id, request.n, request.include_seen
                )
                return MovieRecommendationResponse(
                    recommendations=recommendations,
                    latency_ms=latency
                )
            except Exception as e:
                logging.error(f"Recommendation error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/health")
        async def health_check():
            if self.model is None:
                raise HTTPException(status_code=503, detail="Model not loaded")
            return {"status": "healthy"}
        
        # Start server
        port = self.config.get("service_port", 8000)
        host = self.config.get("service_host", "0.0.0.0")
        logging.info(f"Starting API server at {host}:{port}")
        uvicorn.run(app, host=host, port=port)