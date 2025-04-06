import mlflow.pytorch
import torch
import time

def run_inference():
    # Load the model from MLFlow (example assumes a model was logged under 'model')
    model_uri = "models:/training_experiment/model"
    model = mlflow.pytorch.load_model(model_uri)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Dummy inference: simulate processing a batch
    dummy_input = torch.rand(64, 10).to(device)
    start_time = time.time()
    with torch.no_grad():
        outputs = model(dummy_input)
    latency = (time.time() - start_time) * 1000  # in milliseconds
    print(f"Inference latency: {latency:.2f} ms")
    print(f"Model output: {outputs[:5]}")  # Print first 5 outputs for verification