import pandas as pd
import os

def load_movielens_data():
    # For example, load MovieLens data from CSV files
    data_path = os.path.join("..", "data", "movielens", "ml-latest-small", "ratings.csv")
    try:
        df = pd.read_csv(data_path)
        # For demonstration, we use a subset and only numerical columns
        df = df.select_dtypes(include=['number']).fillna(0)
        # Convert DataFrame to list of records (or any format suitable for training)
        data = df.values.tolist()
        return data
    except Exception as e:
        print("Error loading dataset:", e)
        return []
