import mlflow

def start_mlflow_run(experiment_name):
    mlflow.set_experiment(experiment_name)
    return mlflow.start_run()
