import argparse
from train import run_training
from tune import run_tuning
from inference import run_inference

def main():
    parser = argparse.ArgumentParser(description="MLOps Pipeline for Movie Recommendation System")
    parser.add_argument("--mode", type=str, choices=["train", "tune", "inference"], required=True,
                        help="Mode to run: train, tune, or inference")
    args = parser.parse_args()

    if args.mode == "train":
        run_training()
    elif args.mode == "tune":
        run_tuning()
    elif args.mode == "inference":
        run_inference()

if __name__ == "__main__":
    main()
