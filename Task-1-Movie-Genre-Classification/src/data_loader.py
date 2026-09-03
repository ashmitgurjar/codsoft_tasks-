import os
import glob
import pandas as pd
import numpy as np

KAGGLE_CACHE_DIR = os.path.expanduser("~/.cache/kagglehub/datasets/hijest/genre-classification-dataset-imdb/versions/1/Genre Classification Dataset")

def load_kaggle_dataset(output_path="data/movies_dataset.csv", max_samples_per_genre=1500):
    """
    Loads the official Kaggle 'hijest/genre-classification-dataset-imdb' dataset.
    Extracts train and test splits, cleans formatting, and saves a consolidated benchmark dataset.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    train_file = os.path.join(KAGGLE_CACHE_DIR, "train_data.txt")
    test_solution_file = os.path.join(KAGGLE_CACHE_DIR, "test_data_solution.txt")

    dfs = []
    if os.path.exists(train_file):
        print(f"Reading Kaggle dataset from {train_file}...")
        df_train = pd.read_csv(train_file, sep=" ::: ", engine="python", names=["id", "title", "genre", "description"])
        dfs.append(df_train)

    if os.path.exists(test_solution_file):
        print(f"Reading Kaggle test solution from {test_solution_file}...")
        df_test = pd.read_csv(test_solution_file, sep=" ::: ", engine="python", names=["id", "title", "genre", "description"])
        dfs.append(df_test)

    if not dfs:
        raise FileNotFoundError("Kaggle IMDb dataset files not found.")

    df_full = pd.concat(dfs, ignore_index=True)

    # Clean text columns
    df_full["genre"] = df_full["genre"].astype(str).str.strip().str.title()
    df_full["title"] = df_full["title"].astype(str).str.strip()
    df_full["plot"] = df_full["description"].astype(str).str.strip()

    # Filter out extremely rare genres (< 300 samples) to ensure high statistical stability
    genre_counts = df_full["genre"].value_counts()
    valid_genres = genre_counts[genre_counts >= 300].index.tolist()
    df_filtered = df_full[df_full["genre"].isin(valid_genres)].copy()

    # Balanced sampling per genre to prevent severe class imbalance (e.g. Drama/Documentary overpowering other genres)
    sampled_dfs = []
    for g, group in df_filtered.groupby("genre"):
        if len(group) > max_samples_per_genre:
            sampled_dfs.append(group.sample(n=max_samples_per_genre, random_state=42))
        else:
            sampled_dfs.append(group)

    df_balanced = pd.concat(sampled_dfs, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    df_balanced = df_balanced[["id", "title", "genre", "plot"]]

    df_balanced.to_csv(output_path, index=False)
    print(f"Successfully processed {len(df_balanced)} Kaggle IMDb records across {df_balanced['genre'].nunique()} genres -> Saved to '{output_path}'")
    return df_balanced

def load_dataset(file_path="data/movies_dataset.csv"):
    if not os.path.exists(file_path):
        return load_kaggle_dataset(file_path)
    df = pd.read_csv(file_path)
    return df

if __name__ == "__main__":
    df = load_kaggle_dataset()
    print("\nClass Distribution:")
    print(df["genre"].value_counts())
