"""
MLFlow tracking utilities for experiment logging.
"""

import os
import logging
from typing import Dict, Any, Optional

import mlflow
from mlflow.tracking import MlflowClient


class MLFlowTracker:
    """
    MLFlow tracker for experiment logging.
    """
    
    def __init__(self, experiment_name: str, tracking_uri: Optional[str] = None):
        """
        Initialize MLFlow tracker.
        
        Args:
            experiment_name: Name of the MLFlow experiment
            tracking_uri: URI of the MLFlow tracking server
        """
        self.experiment_name = experiment_name
        
        # Set up tracking URI if provided
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        # Make sure the experiment exists
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            logging.info(f"Creating new experiment: {experiment_name}")
            self.experiment_id = mlflow.create_experiment(experiment_name)
        else:
            self.experiment_id = experiment.experiment_id
            
        self.client = MlflowClient()
        self.active_run = None
        
        logging.info(f"MLFlow tracking initialized for experiment: {experiment_name}")
    
    def start_run(self, run_name: Optional[str] = None) -> mlflow.ActiveRun:
        """
        Start a new MLFlow run.
        
        Args:
            run_name: Optional name for the run
            
        Returns:
            MLFlow active run
        """
        self.active_run = mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name
        )
        logging.info(f"Started MLFlow run: {run_name} (ID: {self.active_run.info.run_id})")
        return self.active_run
    
    def end_run(self) -> None:
        """End the current MLFlow run."""
        if self.active_run:
            mlflow.end_run()
            logging.info(f"Ended MLFlow run: {self.active_run.info.run_id}")
            self.active_run = None
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log parameters to MLFlow.
        
        Args:
            params: Dictionary of parameters to log
        """
        if self.active_run:
            mlflow.log_params(params)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log metrics to MLFlow.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step for the metrics
        """
        if self.active_run:
            mlflow.log_metrics(metrics, step=step)
    
    def log_artifact(self, local_path: str) -> None:
        """
        Log an artifact to MLFlow.
        
        Args:
            local_path: Path to the local file
        """
        if self.active_run:
            mlflow.log_artifact(local_path)
    
    def log_model(self, model: Any, artifact_path: str) -> None:
        """
        Log a model to MLFlow.
        
        Args:
            model: PyTorch model
            artifact_path: Path for the artifact
        """
        if self.active_run:
            mlflow.pytorch.log_model(model, artifact_path)
    
    def set_tag(self, key: str, value: str) -> None:
        """
        Set a tag in MLFlow.
        
        Args:
            key: Tag key
            value: Tag value
        """
        if self.active_run:
            mlflow.set_tag(key, value)