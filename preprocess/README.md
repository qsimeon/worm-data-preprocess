# Module Information

This repository contains the functions used to preprocess the open-source calcium imaging data for neural activity analysis.

## File Structure

The submodule consists of the following files:
- `_config.py`: Contains configuration settings and parameters for preprocessing.
- `_main.py`: Contains the overarching logic for preprocessing the data as specified in the configuration file.
- `_utils.py`: Contains a number of main utility functions for beginning preprocessing.
- `_pkg.py`: Contains the necessary imports for the submodule.
- `preprocess/_base_preprocessors.py`: Contains the base preprocessing classes from which all dataset-specific preprocessors inherit their methods.
- `preprocess/_neural.py`: Contains all the neural-specific preprocessors (unconcerned with connectome data).
- `preprocess/_helpers.py`: Contains helper functions used across different preprocessing classes.


## Usage

Please refer to the `README.md` in the root directory for information on how to
process the data.

### Dataset Structure

Each dataset is stored in a Python dictionary of the following form:
```python
data_dict = {
    "worm0": { ... },
    "worm1": { ... },
    "...":   { ... },
    "wormN": { ... },
}
```

<details>
<summary>Here is information about all the data associated with each worm</summary>
Each worm (`worm0`, `worm1`, ..., `wormN`) is a dictionary containing:

| Column                                | Type            | Description                 |
|---------------------------------------|-----------------|-----------------------------|
| `calcium_data`                        | torch.Tensor    | Normalized, resampled data  |
| `source_dataset`                      | str             | Source dataset name         |
| `dt`                                  | torch.Tensor    | Time deltas (resampled)     |
| `interpolate_method`                  | str             | Interpolation method        |
| `max_timesteps`                       | int             | Timesteps after resampling  |
| `median_dt`                           | float           | Median of resampled dt      |
| `num_labeled_neurons`                 | int             | Count labeled neurons       |
| `num_neurons`                         | int             | Total neuron count          |
| `num_unlabeled_neurons`               | int             | Count unlabeled neurons     |
| `original_dt`                         | torch.Tensor    | Original time deltas        |
| `original_calcium_data`               | torch.Tensor    | Raw calcium data            |
| `normalization_method`                | str             | Normalization method        |
| `original_max_timesteps`              | int             | Timesteps before resampling |
| `original_median_dt`                  | float           | Median original dt          |
| `original_residual_calcium`           | torch.Tensor    | Original residual data      |
| `original_smooth_calcium_data`        | torch.Tensor    | Smoothed original data      |
| `original_smooth_residual_calcium`    | torch.Tensor    | Smoothed original residuals |
| `original_time_in_seconds`            | torch.Tensor    | Original timestamps         |
| `residual_calcium`                    | torch.Tensor    | Residual calcium data       |
| `smooth_calcium_data`                 | torch.Tensor    | Smoothed calcium data       |
| `smooth_method`                       | str             | Smoothing method            |
| `smooth_residual_calcium`             | torch.Tensor    | Smoothed residual data      |
| `time_in_seconds`                     | torch.Tensor    | Resampled timestamps        |
| `worm`                                | str             | Worm identifier             |
| `extra_info`                          | dict            | Additional metadata         |
| `labeled_neuron_to_slot`              | dict            | Labeled neuron → index      |
| `labeled_neurons_mask`                | torch.Tensor    | Mask for labeled neurons    |
| `neuron_to_slot`                      | dict            | Neuron → index mapping      |
| `neurons_mask`                        | torch.Tensor    | Mask for all neurons        |
| `slot_to_labeled_neuron`              | dict            | Index → labeled neuron      |
| `slot_to_neuron`                      | dict            | Index → neuron mapping      |
| `slot_to_unlabeled_neuron`            | dict            | Index → unlabeled neuron    |
| `unlabeled_neuron_to_slot`            | dict            | Unlabeled neuron → index    |
| `unlabeled_neurons_mask`              | torch.Tensor    | Mask for unlabeled neurons  |

</details>

## Preprocessing

The datasets have been preprocessed using Python scripts available in this repository. The preprocessing steps include:

1. **Loading** raw data in various formats (MATLAB files, JSON files, etc.).
1. Extracting relevant data fields (neuron IDs, traces, time vectors, etc.).
1. Cleaning data
1. **Resampling** the data to a common time resolution. - if requested
1. **Smoothing** the data using different methods - if requested
1. **Normalizing** data
1. Creating dictionaries to map neuron indices to neuron IDs and vice versa.
1. Saving the preprocessed data into a standardized format.


