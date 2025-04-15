# ML Operations and Systems Engineering

This repository implements a production-grade, cloud-native movie recommendation system with an end-to-end MLOps pipeline.

```
project-root/
├── .github/
│   └── workflows/
│       └── ci-cd.yml
├── configs/
│   ├── train_config.yaml
│   ├── tune_config.yaml
│   └── infer_config.yaml
├── data/
│   ├── download_data.sh
│   └── movielens/
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yaml
├── infrastructure/
│   ├── main.tf
│   ├── variables.tf
│   └── db/
│       ├── init.sql
│       └── init.sh
├── k8s/
│   ├── deployment.yaml
│   ├── ingress.yaml
│   ├── service.yaml
│   ├── secrets.yaml
│   └── pvc.yaml
├── models/
├── monitoring/
│   ├── grafana_dashboard.json
│   └── prometheus.yml
├── src/
│   ├── main.py
│   ├── mlflow_tracking.py
│   ├── models.py
│   ├── train.py (you have this)
│   ├── tune.py (you have this)
│   ├── inference.py (you have this)
│   ├── utils.py
│   └── requirements.txt
├── tests/
├── .gitignore
├── LICENSE
├── readme.md
├── requirements.txt
└── setup-directories.sh
```

## Key Features
- **MLFlow** for experiment tracking and model versioning.
- **Terraform** for AWS EC2 infrastructure provisioning.
- **Docker & Kubernetes** for containerization and scalable inference services.
- **Ray** for distributed hyperparameter tuning.
- **PyTorch** for multi-GPU training of the recommendation model.
- **Prometheus & Grafana** for real-time monitoring.
- **CI/CD** practices via GitHub Actions for continuous experimentation and deployment.
- **Inference benchmarking** with custom Triton kernel integration to assess speed under different quantization levels and batch sizes.

## Getting Started

### Prerequisites
- AWS account with proper credentials
- Docker & Kubernetes (e.g., Minikube for local testing)
- Terraform installed
- Python 3.8+

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/ML-Operations-and-Systems-Engineering.git
   cd ML-Operations-and-Systems-Engineering

    ```

2. Download Datasets:
   ```bash
   cd data
   bash download_data.sh
   cd ..

    ```

3. **Provision Infrastructure with Terraform:** Navigate to the infrastructure folder and initialize/apply:

    ```bash
    cd infrastructure
    terraform init
    terraform apply
    cd ..
    ```


4. **Build Docker Images:** Navigate to the `docker` folder and build the images:

    ```bash
    cd docker
    docker build -t movie-recommender:latest .
    docker build -t movie-recommender-inference:latest -f Dockerfile.inference .
    cd ..
    ```

5. **Run Training, Tuning, or Inference: The main script (src/main.py) serves as a CLI to run various stages:**

    ```bash
    python src/main.py --stage train --config configs/train_config.yaml
    python src/main.py --stage tune --config configs/tune_config.yaml
    python src/main.py --stage infer --config configs/infer_config.yaml
    ```
6. **Deploying to Kubernetes:** Use the provided Kubernetes YAML files in the `k8s` folder to deploy the inference service.

    ```bash
    kubectl apply -f k8s/deployment.yaml
    kubectl apply -f k8s/service.yaml
    ```

7. **Monitoring:** Set up Prometheus and Grafana using the provided Helm charts or YAML files in the `monitoring` folder.

    ```bash
    cd monitoring
    helm install prometheus prometheus-community/prometheus
    helm install grafana grafana/grafana
    cd ..
    ```

8. **CI/CD:** GitHub Actions workflows are defined in `.github/workflows/`. Modify them as per your requirements.
   - **Continuous Experimentation:** Automatically trigger experiments on code changes or new data.
   - **Continuous Deployment:** Deploy the latest model to production after successful tests.
    - **Continuous Monitoring:** Monitor model performance and system metrics in real-time.
    - **Continuous Feedback Loop:** Use monitoring data to retrain the model periodically.
    