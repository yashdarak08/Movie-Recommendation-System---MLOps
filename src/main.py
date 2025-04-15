#!/usr/bin/env python
"""
Main entry point for the movie recommendation system.
"""

import argparse
import logging
import os
import sys
from typing import Dict, Any

import yaml

from mlflow_tracking import MLFlowTracker
from train import MovieRecTrainer
from tune import MovieRecTuner
from inference import MovieRecInference
from utils import setup_logging


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Movie Recommendation System")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "tune", "inference", "serve"],
        help="Mode to run the system in"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to the configuration file"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the trained model (for inference mode)"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/movielens/ml-latest-small",
        help="Path to the dataset"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level"
    )
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from yaml file."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logging.error(f"Error parsing configuration file: {e}")
        sys.exit(1)


def main():
    """Main entry point."""
    args = parse_args()
    setup_logging(args.log_level)
    
    # Load configuration
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        logging.warning(f"Config file {args.config} not found. Using default parameters.")
        config = {}
    
    # Update config with command line arguments
    config["data_path"] = args.data_path
    if args.model_path:
        config["model_path"] = args.model_path
    
    # Initialize MLFlow tracking
    mlflow_tracker = MLFlowTracker(
        experiment_name=config.get("experiment_name", "movie_recommendation"),
        tracking_uri=config.get("mlflow_tracking_uri", "http://localhost:5000")
    )
    
    # Run the specified mode
    if args.mode == "train":
        logging.info("Starting training mode")
        trainer = MovieRecTrainer(config, mlflow_tracker)
        trainer.train()
        
    elif args.mode == "tune":
        logging.info("Starting hyperparameter tuning mode")
        tuner = MovieRecTuner(config, mlflow_tracker)
        tuner.tune()
        
    elif args.mode == "inference":
        logging.info("Starting inference mode")
        inference = MovieRecInference(config)
        inference.run_inference()
        
    elif args.mode == "serve":
        logging.info("Starting serving mode")
        inference = MovieRecInference(config)
        inference.start_service()
    
    logging.info(f"Completed {args.mode} mode")


if __name__ == "__main__":
    main()