import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import mlflow
from mlflow_tracking import start_mlflow_run
from utils import load_movielens_data

# Dummy Dataset and Model for illustration purposes
class MovieLensDataset(Dataset):
    def __init__(self, data):
        self.data = data  # data should be a list or DataFrame

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Replace with actual data retrieval logic
        sample = self.data[idx]
        return sample

class RecSysModel(nn.Module):
    def __init__(self, num_features, hidden_size=128):
        super(RecSysModel, self).__init__()
        self.fc1 = nn.Linear(num_features, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)  # Predict a rating or score

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)

def run_training():
    # Load data (for example purposes, using dummy data)
    data = load_movielens_data()
    dataset = MovieLensDataset(data)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Initialize model, loss and optimizer
    model = RecSysModel(num_features=10)  # Replace 10 with actual number of features
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)  # Multi-GPU support
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Start MLFlow run
    with start_mlflow_run("training_experiment"):
        for epoch in range(5):  # Replace with desired number of epochs
            model.train()
            running_loss = 0.0
            for batch in dataloader:
                inputs = batch.float().to(device)  # adjust according to your dataset
                targets = torch.rand(inputs.size(0), 1).to(device)  # dummy target; replace with actual labels
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            avg_loss = running_loss / len(dataloader)
            mlflow.log_metric("loss", avg_loss, step=epoch)
            print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
        # Optionally save the model
        mlflow.pytorch.log_model(model, "model")
        print("Model saved to MLflow")
        mlflow.log_artifact("model/model.pth", artifact_path="models")