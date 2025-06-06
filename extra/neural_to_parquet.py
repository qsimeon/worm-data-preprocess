"""
Converts preprocessed neural dataset files into a parquet file that can be 
uploaded and viewed on HuggingFace.
"""

import os
import numpy as np
import pandas as pd
import pickle

from tqdm import tqdm 
# import matplotlib.pyplot as plt

# for imports to work even in /extra directory
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from _main_utils import (
    init_random_seeds,
    ROOT_DIR
)

from preprocess.config import PREPROCESS_CONFIG
EXPERIMENT_DATASETS = PREPROCESS_CONFIG["EXPERIMENT_DATASETS"]


def all_zeros(lst):
    """Check if all values in a list are 0.0."""
    return all(val == 0.0 for val in lst)

def load_dataset(name):
    """Load a specified neural dataset's pickle file by name"""
    assert (name in EXPERIMENT_DATASETS), "Unrecognized dataset!"
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

    desired_columns = {
        "source_dataset",
        "raw_data_file",
        "worm",
        "neuron",
        "slot",
        "is_labeled_neuron",
        "smooth_method",
        "interpolate_method",  
        "normalization_method", # Standard or Causal
        "original_calcium_data",
        "calcium_data",
        "original_time_in_seconds",
        "time_in_seconds",
        "original_max_timesteps",
        "max_timesteps",
    }

    for dataset in progress_bar:
        for worm in dataset:
            progress_bar.set_description(
                f"Processing {dataset[worm]['source_dataset']}"
            )
            for neuron, slot in dataset[worm]["neuron_to_slot"].items():
                slot = np.intc(slot)
                dataset_name = dataset[worm]["source_dataset"]
                raw_data_file = dataset[worm]["extra_info"]["data_file"]
                smooth_method = dataset[worm]["smooth_method"]
                smooth_method = (
                    "no smoothing" if smooth_method is None else smooth_method.lower()
                )
                interpolate_method = dataset[worm]["interpolate_method"]

                original_calcium_data = (
                    dataset[worm]["original_calcium_data"][:, slot]
                    .numpy()
                    .astype(np.float32)
                )
                calcium_data = (
                    dataset[worm]["calcium_data"][:, slot].numpy().astype(np.float32)
                )

                normalization_method = dataset[worm]["normalization_method"]
                if normalization_method == "causal":
                    cumulative_mean = (
                        dataset[worm]["cumulative_mean"][:, slot]
                        .numpy()
                        .astype(np.float32)
                    )
                    cumulative_std = (
                        dataset[worm]["cumulative_std"][:, slot]
                        .numpy()
                        .astype(np.float32)
                    )

                # remove worm if calcium_data is all zero
                if all_zeros(calcium_data):
                    continue

                original_smooth_calcium_data = (
                    dataset[worm]["original_smooth_calcium_data"][:, slot]
                    .numpy()
                    .astype(np.float32)
                )
                smooth_calcium_data = (
                    dataset[worm]["smooth_calcium_data"][:, slot]
                    .numpy()
                    .astype(np.float32)
                )
                original_residual_calcium = (
                    dataset[worm]["original_residual_calcium"][:, slot]
                    .numpy()
                    .astype(np.float32)
                )
                residual_calcium = (
                    dataset[worm]["residual_calcium"][:, slot]
                    .numpy()
                    .astype(np.float32)
                )
                original_smooth_residual_calcium = (
                    dataset[worm]["original_smooth_residual_calcium"][:, slot]
                    .numpy()
                    .astype(np.float32)
                )
                smooth_residual_calcium = (
                    dataset[worm]["smooth_residual_calcium"][:, slot]
                    .numpy()
                    .astype(np.float32)
                )

                original_time_in_seconds = (
                    dataset[worm]["original_time_in_seconds"]
                    .squeeze()
                    .numpy()
                    .astype(np.float32)
                )
                time_in_seconds = (
                    dataset[worm]["time_in_seconds"]
                    .squeeze()
                    .numpy()
                    .astype(np.float32)
                )
                original_dt = (
                    dataset[worm]["original_dt"].squeeze().numpy().astype(np.float32)
                )
                dt = dataset[worm]["dt"].squeeze().numpy().astype(np.float32)
                original_median_dt = np.float32(dataset[worm]["original_median_dt"])
                median_dt = np.float32(dataset[worm]["median_dt"])
                original_max_timesteps = np.intc(
                    dataset[worm]["original_max_timesteps"]
                )
                max_timesteps = np.intc(dataset[worm]["max_timesteps"])

                is_labeled_neuron = neuron in dataset[worm]["labeled_neuron_to_slot"]

                # all the columns from `preprocess_traces` worm dict in NeuralBaseProcessor
                all_columns = {
                    "source_dataset": dataset_name,
                    "raw_data_file": raw_data_file,
                    "worm": worm,
                    "neuron": neuron,
                    "slot": slot,
                    "is_labeled_neuron": is_labeled_neuron,
                    "smooth_method": smooth_method,
                    "interpolate_method": interpolate_method,
                    "normalization_method": normalization_method,
                    "original_calcium_data": original_calcium_data,
                    "calcium_data": calcium_data,
                    "cumulative_mean": cumulative_mean
                    if "cumulative_mean" in dataset[worm]
                    else None,
                    
                    "cumulative_std": cumulative_std
                    if "cumulative_std" in dataset[worm]
                    else None,
                    
                    "original_smooth_calcium_data": original_smooth_calcium_data,
                    "smooth_calcium_data": smooth_calcium_data,
                    "original_residual_calcium": original_residual_calcium,
                    "residual_calcium": residual_calcium,
                    "original_smooth_residual_calcium": original_smooth_residual_calcium,
                    "smooth_residual_calcium": smooth_residual_calcium,
                    "original_time_in_seconds": original_time_in_seconds,
                    "time_in_seconds": time_in_seconds,
                    "original_dt": original_dt,
                    "dt": dt,
                    "original_median_dt": original_median_dt,
                    "median_dt": median_dt,
                    "original_max_timesteps": original_max_timesteps,
                    "max_timesteps": max_timesteps,
                }

                for column, value in all_columns.items():
                    if column in desired_columns and value is not None:
                        data_dict.setdefault(column, []).append(value)

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
    # for testing:
    # datasets = load_all_worm_datasets()
    # df = process_worm_datasets(datasets)
    # return
    
    init_random_seeds(42)
    
    # Prompt user for file name
    name = input("What would you like to name your saved dataset? processed_worm_data_").strip()
    if not name:
        print("Invalid file name. Exiting.")
        return

    # Prompt user for file format
    save_option = input(
        "Select file format to save: \n'0' for .parquet \n'1' for .csv \n'2' for .both: "
    ).strip()
    if save_option not in {"0", "1", "2"}:
        print("Invalid option. Exiting.")
        return

    datasets = load_all_worm_datasets()
    df = process_worm_datasets(datasets)

    # Save to parquet and/or csv (for viewing locally)
    os.makedirs(os.path.join(ROOT_DIR, "datasets"), exist_ok=True)
    parquet_filename = f"processed_worm_data_{name}.parquet"
    csv_filename = f"processed_worm_data_{name}.csv"
    parquet_path = os.path.join(ROOT_DIR, "datasets", parquet_filename)
    csv_path = os.path.join(ROOT_DIR, "datasets", csv_filename)

    if os.path.exists(parquet_path) or os.path.exists(csv_path):
        overwrite = (
            input(
                f"File datasets/{parquet_filename} or datasets/{csv_filename} already exists. Would you like to overwrite it? (yes/no): "
            )
            .strip()
            .lower()
        )
        if overwrite != "yes":
            return

    if save_option in {"0", "2"}:
        print(f"Saving processed data to datasets/{parquet_filename}...")
        df.to_parquet(parquet_path, index=False, engine="pyarrow")

    if save_option in {"1", "2"}:
        print(f"Saving processed data to datasets/{csv_filename}...")
        df.to_csv(csv_path, index=False)

    print("Processed data saved.")

    # Visualize the dataset (requires matplotlib)
    # visualize_data(df)

if __name__ == "__main__":
    main()
