"""
Converts preprocessed neural dataset files into a parquet file that can be 
uploaded and viewed on HuggingFace.
"""

import os
import torch
import scipy.stats
import numpy as np
import pandas as pd
import pickle

from tqdm import tqdm 
# import matplotlib.pyplot as plt

from _main_utils import (
    init_random_seeds,
    SYNTHETIC_DATASETS,
    ROOT_DIR
)

from preprocess.config import PREPROCESS_CONFIG
EXPERIMENT_DATASETS = PREPROCESS_CONFIG["EXPERIMENT_DATASETS"]


def all_zeros(lst):
    """Check if all values in a list are 0.0."""
    return all(val == 0.0 for val in lst)

def load_dataset(name):
    """Load a specified neural dataset's pickle file by name"""
    assert (name in EXPERIMENT_DATASETS) or (name in SYNTHETIC_DATASETS), "Unrecognized dataset!"
    file = os.path.join(ROOT_DIR, "data", "processed", "neural", f"{name}.pickle")
    assert os.path.exists(file), f"The file {file} does not exist."
    with open(file, "rb") as pickle_in:
        return pickle.load(pickle_in)

def load_all_worm_datasets():
    """Load all neural datasets and return a list of them"""
    print("Loading all worm datasets...")
    return [load_dataset(dataset) for dataset in EXPERIMENT_DATASETS]

def process_worm_datasets(datasets):
    """Process worm datasets and return a DataFrame with extracted information"""
    print("Processing worm datasets...")
    data_dict = {}
    progress_bar = tqdm(datasets)

    for dataset in progress_bar:
        for worm in dataset:
            progress_bar.set_description(f"Processing {dataset[worm]['source_dataset']}")
            for neuron, slot in dataset[worm]["neuron_to_slot"].items():
                slot = np.intc(slot)
                dataset_name = dataset[worm]["source_dataset"]
                raw_data_file = dataset[worm]["extra_info"]["data_file"]
                # TODO: All smooth methods are currently "none" 
                smooth_method = dataset[worm]["smooth_method"]
                smooth_method = "no smoothing" if smooth_method is None else smooth_method.lower()
                interpolate_method = dataset[worm]["interpolate_method"]

                original_calcium_data = dataset[worm]["original_calcium_data"][:, slot].numpy().astype(np.float32)
                calcium_data = dataset[worm]["calcium_data"][:, slot].numpy().astype(np.float32)

                if all_zeros(calcium_data):
                    continue

                original_smooth_calcium_data = dataset[worm]["original_smooth_calcium_data"][:, slot].numpy().astype(np.float32)
                smooth_calcium_data = dataset[worm]["smooth_calcium_data"][:, slot].numpy().astype(np.float32)
                original_residual_calcium = dataset[worm]["original_residual_calcium"][:, slot].numpy().astype(np.float32)
                residual_calcium = dataset[worm]["residual_calcium"][:, slot].numpy().astype(np.float32)
                original_smooth_residual_calcium = dataset[worm]["original_smooth_residual_calcium"][:, slot].numpy().astype(np.float32)
                smooth_residual_calcium = dataset[worm]["smooth_residual_calcium"][:, slot].numpy().astype(np.float32)

                original_time_in_seconds = dataset[worm]["original_time_in_seconds"].squeeze().numpy().astype(np.float32)
                time_in_seconds = dataset[worm]["time_in_seconds"].squeeze().numpy().astype(np.float32)
                original_dt = dataset[worm]["original_dt"].squeeze().numpy().astype(np.float32)
                dt = dataset[worm]["dt"].squeeze().numpy().astype(np.float32)
                original_median_dt = np.float32(dataset[worm]["original_median_dt"])
                median_dt = np.float32(dataset[worm]["median_dt"])
                original_max_timesteps = np.intc(dataset[worm]["original_max_timesteps"])
                max_timesteps = np.intc(dataset[worm]["max_timesteps"])

                is_labeled_neuron = neuron in dataset[worm]["labeled_neuron_to_slot"]
                is_unlabeled_neuron = neuron in dataset[worm]["unlabeled_neuron_to_slot"]

                data_dict.setdefault("source_dataset", []).append(dataset_name)
                data_dict.setdefault("raw_data_file", []).append(raw_data_file)
                data_dict.setdefault("worm", []).append(worm)
                data_dict.setdefault("neuron", []).append(neuron)
                data_dict.setdefault("slot", []).append(slot)
                data_dict.setdefault("is_labeled_neuron", []).append(is_labeled_neuron)
                data_dict.setdefault("is_unlabeled_neuron", []).append(is_unlabeled_neuron)
                data_dict.setdefault("original_calcium_data", []).append(original_calcium_data)
                data_dict.setdefault("calcium_data", []).append(calcium_data)
                data_dict.setdefault("original_smooth_calcium_data", []).append(original_smooth_calcium_data)
                data_dict.setdefault("smooth_calcium_data", []).append(smooth_calcium_data)
                data_dict.setdefault("original_residual_calcium", []).append(original_residual_calcium)
                data_dict.setdefault("residual_calcium", []).append(residual_calcium)
                data_dict.setdefault("original_smooth_residual_calcium", []).append(original_smooth_residual_calcium)
                data_dict.setdefault("smooth_residual_calcium", []).append(smooth_residual_calcium)
                data_dict.setdefault("smooth_method", []).append(smooth_method)
                data_dict.setdefault("interpolate_method", []).append(interpolate_method)
                data_dict.setdefault("original_time_in_seconds", []).append(original_time_in_seconds)
                data_dict.setdefault("time_in_seconds", []).append(time_in_seconds)
                data_dict.setdefault("original_dt", []).append(original_dt)
                data_dict.setdefault("dt", []).append(dt)
                data_dict.setdefault("original_median_dt", []).append(original_median_dt)
                data_dict.setdefault("median_dt", []).append(median_dt)
                data_dict.setdefault("original_max_timesteps", []).append(original_max_timesteps)
                data_dict.setdefault("max_timesteps", []).append(max_timesteps)

    df = pd.DataFrame.from_dict(data_dict)
    print("Processing complete.")
    return df

# # Requires matplotlib to be installed
# def visualize_data(df):
#     """Generate a heatmap visualization of the processed dataset"""
#     plt.figure(figsize=(10, 6))
#     sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
#     plt.title("Correlation Heatmap of Worm Dataset")
#     plt.show()

def main():
    init_random_seeds(42)
    
    datasets = load_all_worm_datasets()
    df = process_worm_datasets(datasets)
    
    # # prints only "none" for all 42811 rows 
    # print(f"Number of rows in the dataframe: {len(df)}")
    # print("Most frequently occurring smooth methods:")
    # print(df['smooth_method'].value_counts().head())

    # Save to parquet
    os.makedirs(os.path.join(ROOT_DIR, "datasets"), exist_ok=True)
    parquet_filename = "processed_worm_data_short.parquet"
    parquet_path = os.path.join(ROOT_DIR, "datasets", parquet_filename)
    if not os.path.exists(parquet_path):
        print(f"Saving processed data to datasets/{parquet_filename}...")
        df.to_parquet(parquet_path, index=False, engine="pyarrow")
        print(f"Processed data saved to datasets/{parquet_filename}")
    else:
        print(f"File datasets/{parquet_filename} already exists. Skipping save.")
    
    # Visualize the dataset (requires matplotlib)
    # visualize_data(df)

if __name__ == "__main__":
    main()
