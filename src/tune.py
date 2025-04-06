import mlflow
from ray import tune
from ray.tune.integration.mlflow import MLflowLoggerCallback
from train import RecSysModel  # Assuming the model is defined here
import torch

def train_func(config):
    # Dummy training loop for hyperparameter tuning
    # In practice, load your data and build your training loop here
    model = RecSysModel(num_features=10, hidden_size=config["hidden_size"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Dummy loss computation
    dummy_loss = (config["hidden_size"] - 128)**2 / 1000.0
    tune.report(loss=dummy_loss)

def run_tuning():
    mlflow_callback = MLflowLoggerCallback(experiment_name="hyperparam_tuning")
    analysis = tune.run(
        train_func,
        config={
            "hidden_size": tune.grid_search([64, 128, 256])
        },
        num_samples=1,
        callbacks=[mlflow_callback]
    )
    print("Best config: ", analysis.get_best_config(metric="loss", mode="min"))
    print("Best trial final validation loss: ", analysis.best_trial.last_result["loss"])
    print("Best trial ID: ", analysis.best_trial.trial_id)
    print("Best trial config: ", analysis.best_trial.config)