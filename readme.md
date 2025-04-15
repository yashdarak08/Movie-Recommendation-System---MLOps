# MLOps Movie Recommendation System

A production-grade, cloud-native movie recommendation system with an end-to-end MLOps pipeline. This system leverages PyTorch for training recommendation models, MLflow for experiment tracking, and provides scalable inference through Docker and Kubernetes.

## Features

- **Recommendation Models**: Matrix Factorization, Neural Collaborative Filtering, and Transformer-based
- **MLOps Pipeline**: End-to-end workflow from data preparation to deployment
- **Experiment Tracking**: Comprehensive tracking with MLflow
- **Hyperparameter Optimization**: Distributed tuning with Ray
- **Scalable Inference**: REST API for real-time recommendations
- **Containerization**: Docker and Kubernetes deployment
- **Infrastructure as Code**: AWS EC2 provisioning via Terraform
- **CI/CD**: Continuous integration and deployment with GitHub Actions
- **Monitoring**: Real-time metrics with Prometheus and Grafana

## Project Structure

```
project-root/
├── .github/
│   └── workflows/
│       └── ci-cd.yml          # CI/CD workflow definition
├── configs/
│   ├── train_config.yaml      # Training configuration
│   ├── tune_config.yaml       # Hyperparameter tuning configuration
│   └── infer_config.yaml      # Inference configuration
├── data/
│   ├── download_data.sh       # Script to download MovieLens datasets
│   └── movielens/             # MovieLens datasets (after download)
├── docker/
│   ├── Dockerfile             # Main Dockerfile for the application
│   └── docker-compose.yaml    # Docker Compose configuration
├── infrastructure/
│   ├── main.tf                # Terraform main configuration
│   ├── variables.tf           # Terraform variables
│   └── db/
│       ├── init.sql           # Database initialization script
│       └── init.sh            # Database init script wrapper
├── k8s/
│   ├── deployment.yaml        # Kubernetes deployment configuration
│   ├── ingress.yaml           # Kubernetes ingress configuration
│   ├── service.yaml           # Kubernetes service configuration
│   ├── secrets.yaml           # Kubernetes secrets configuration
│   └── pvc.yaml               # Kubernetes persistent volume claim
├── models/                    # Directory for saved models
├── monitoring/
│   ├── grafana_dashboard.json # Grafana dashboard configuration
│   └── prometheus.yml         # Prometheus configuration
├── scripts/
│   └── test_api.py            # Script to test the API endpoints
├── src/
│   ├── main.py                # Main entry point
│   ├── mlflow_tracking.py     # MLflow tracking utilities
│   ├── models.py              # Model definitions
│   ├── train.py               # Training module
│   ├── tune.py                # Hyperparameter tuning module
│   ├── inference.py           # Inference and serving module
│   ├── utils.py               # Utility functions
│   └── requirements.txt       # Python dependencies
├── tests/
│   ├── unit/                  # Unit tests
│   │   └── test_models.py     # Tests for models
│   └── integration/           # Integration tests
│       └── test_inference.py  # Tests for inference API
├── .gitignore                 # Git ignore file
├── LICENSE                    # MIT License
├── README.md                  # This file
├── requirements.txt           # Top-level Python dependencies
└── setup-directories.sh       # Script to set up directory structure
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Docker and Docker Compose
- Kubernetes (optional, for production deployment)
- Terraform (optional, for cloud deployment)
- CUDA-compatible GPU (recommended for training)

### Setting Up the Environment

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/mlops-movie-recommendation.git
   cd mlops-movie-recommendation
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download and prepare data:
   ```bash
   cd data
   bash download_data.sh
   cd ..
   ```

5. Set up directory structure (if needed):
   ```bash
   bash setup-directories.sh
   ```

## Usage

### Training a Model

1. Review and adjust the training configuration in `configs/train_config.yaml`

2. Run the training:
   ```bash
   python -m src.main --mode train --config configs/train_config.yaml
   ```

3. The trained model will be saved in the `models/` directory and tracked in MLflow

### Hyperparameter Tuning

1. Review and adjust the tuning configuration in `configs/tune_config.yaml`

2. Run the hyperparameter tuning:
   ```bash
   python -m src.main --mode tune --config configs/tune_config.yaml
   ```

3. The best model will be saved to the `models/` directory

### Running Inference

1. Start the inference server:
   ```bash
   python -m src.inference --mode serve --config configs/infer_config.yaml
   ```

2. The API will be available at `http://localhost:8000`

3. You can test the API using the provided test script:
   ```bash
   python scripts/test_api.py --url http://localhost:8000
   ```

### Testing the Inference API

The API exposes the following endpoints:

- `GET /health` - Health check endpoint
- `POST /predict` - Predict rating for a single user-item pair
- `POST /batch_predict` - Predict ratings for multiple user-item pairs
- `POST /recommend` - Get personalized recommendations for a user

Example API request with curl:
```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "top_k": 5}'
```

### Benchmarking Inference Performance

To benchmark the inference performance with different batch sizes:

```bash
python -m src.inference --mode benchmark --config configs/infer_config.yaml
```

Results will be saved to `benchmark_results.json`.

## Deployment

### Using Docker Compose

1. Build and start the containers:
   ```bash
   cd docker
   docker-compose up -d
   ```

2. The services will be available at:
   - Recommendation API: http://localhost:8000
   - MLflow: http://localhost:5000
   - Prometheus: http://localhost:9090
   - Grafana: http://localhost:3000

### Deploying to Kubernetes

1. Apply the Kubernetes configurations:
   ```bash
   kubectl apply -f k8s/pvc.yaml
   kubectl apply -f k8s/secrets.yaml
   kubectl apply -f k8s/deployment.yaml
   kubectl apply -f k8s/service.yaml
   kubectl apply -f k8s/ingress.yaml
   ```

2. Check the deployment status:
   ```bash
   kubectl get pods
   kubectl get services
   ```

### Deploying to AWS with Terraform

1. Initialize Terraform:
   ```bash
   cd infrastructure
   terraform init
   ```

2. Apply the Terraform configuration:
   ```bash
   terraform apply
   ```

3. After deployment, Terraform will output the EC2 instance IP address.

## CI/CD Pipeline

The repository includes a GitHub Actions workflow in `.github/workflows/ci-cd.yml` that:

1. Runs linting and tests
2. Builds a Docker image
3. Trains and evaluates models (on selected branches)
4. Deploys to development or production environments (manual trigger)

## Monitoring

The system includes monitoring with Prometheus and Grafana:

1. Prometheus metrics are exposed at `http://localhost:8001/metrics`
2. Grafana dashboard is available at `http://localhost:3000`
   - Default credentials: admin/admin

## License

This project is licensed under the MIT License - see the LICENSE file for details.