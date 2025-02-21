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

TODO:

## Datasets

The following datasets are included in this submodule:

- `Kato2015`: Calcium imaging data from **12** *C. elegans* individuals.
- `Nichols2017`: Calcium imaging data from **44** *C. elegans* individuals.
- `Skora2018`: Calcium imaging data from **12** *C. elegans* individuals.
- `Kaplan2020`: Calcium imaging data from **19** *C. elegans* individuals.
- `Uzel2022`: Calcium imaging data from **6** *C. elegans* individuals.
- `Leifer2023`: Calcium imaging data from **41** *C. elegans* individuals.
- `Flavell2023`: Calcium imaging data from **10** *C. elegans* individuals.

TODO: more than this

### Dataset Structure

TODO: ensure these are up to date

Each dataset is stored in a Python dictionary: 

<details>
<summary>Here you will find its list of features</summary>

- `source_dataset`: (str) Name of the dataset
- `smooth_method`: (str) Method used to smooth the calcium data
- `interpolate_method`: (std) Method used to interpolate the calcium data
- `worm`: (str) The worm ID in the COMBINED dataset (if you load more than one dataset)
- `original_worm`: (str) The worm ID in the original dataset (when you load a single dataset)
- `original_max_timesteps`: (int) Number of time steps before resampling
- `max_timesteps`: (int) Number of time steps after resampling
- `original_dt`: (torch.tensor) Column vector containing the difference between time steps (before resampling). Shape: (original_max_timesteps, 1)
- `original_median_dt`: (float) The median `dt` of the original time series
- `dt`: (torch.tensor) Column vector containing the difference between time steps. Shape: (max_timesteps, 1)
- `residual_` and `original_calcium_data`: (torch.tensor) Standardized and normalized calcium data. Shape: (original_max_timesteps, `NUM_NEURONS`)
- `residual_` and `calcium_data`: (torch.tensor) Standardized, normalized and resampled calcium data. Shape: (max_timesteps, `NUM_NEURONS`)
- `residual_` and `original_smooth_calcium_data`: (torch.tensor) Standardized, smoothed and normalized calcium data. Shape: (original_max_timesteps, `NUM_NEURONS`)
- `residual_` and `smooth_calcium_data`: (torch.tensor) Standardized, smoothed, normalized and resampled calcium data. Shape: (max_timesteps, `NUM_NEURONS`)
- `original_time_in_seconds`: (torch.tensor) A column vector with the original time recording times (without resampling). Shape: (original_max_timesteps, 1)
- `time_in_seconds`: (torch.tensor) A column vector equally spaced by dt after resampling. Shape: (max_timesteps, 1)
- `num_neurons`: (int) Number of total tracked neurons of this specific worm
- `num_labeled_neurons`: (int) Number of labeled neurons
- `num_unlabeled_neurons`: (int) Number of unlabeled neurons
- `labeled_neurons_mask`: (torch.tensor) A bool vector indicating the positions of the labeled neurons. Shape: (`NUM_NEURONS`)
- `unlabeled_neurons_mask`: (torch.tensor) A bool vector indicating the positions of the unlabeled neurons. Shape: (`NUM_NEURONS`)
- `neurons_mask`: (torch.tensor) A bool vector indicating the positions of all tracked neurons (labeled + unlabeled). Shape: (`NUM_NEURONS`)
- `slot_to_labeled_neuron`: (dict) Mapping of column index -> `NUM_NEURONS` neurons. Len: num_neurons
- `labeled_neuron_to_slot`: (dict) Mapping of `NUM_NEURONS` neurons -> column index. Len: num_neurons
- `slot_to_unlabeled_neuron`: (dict) Mapping of column index -> unlabeled neuron. Len: num_unlabeled_neurons
- `unlabeled_neuron_to_slot`: (dict) Mapping of unlabeled neurons -> column index. Len: num_unlabeled_neurons
- `slot_to_neuron`: (dict) Mapping of column index -> labeled+unlabeled neurons. Len: num_neurons
- `neuron_to_slot`: (dict) Mapping of labeled+unlabeled neurons -> column index. Len: num_neurons

</details>

## Preprocessing

The datasets have been preprocessed using Python scripts available in this repository. The preprocessing steps include:

- Loading raw data in various formats (MATLAB files, JSON files, etc.).
- Extracting relevant data fields (neuron IDs, traces, time vectors, etc.).
- Cleaning data
- Resampling the data to a common time resolution. - if requested
- Smoothing the data using different methods - if requested
- Normalizing data
- Creating dictionaries to map neuron indices to neuron IDs and vice versa.
- Saving the preprocessed data in a standardized format.


