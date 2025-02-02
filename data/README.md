# Data Submodule

This submodule contains the code for handling and loading various datasets used in the project.

## File Structure

The submodule consists of the following files:

- `_main.py`: Contains the main function `get_datasets` for retrieving or generating training and validation datasets based on the specified configuration. This function can load datasets if they are provided in a specific directory or generate them from the requested experimental datasets (see the configs submodule for more details).
- `_utils.py`: Contains utility functions and classes for data processing and dataset loading.
- `_pkg.py`: Contains the necessary imports for the submodule.

## Usage

To use the submodule, follow these steps:

1. Install the required Python dependencies as explained in the `setup` folder.
2. Modify the dataset configuration file `configs/submodule/data.yaml` to specify the experimental dataset names and other parameters.
3. Run the `python main.py +submodule=dataset` to load the dataset and obtain the required data. It will create a `dataset` folder inside the `logs` directory, containig the train and validation datasets, the combined dataset before splitting the data into train and validation, and some additional information of them.
4. For more usage examples, see the configuration submodule.

Note: Make sure to have the required dataset files in the appropriate directories before running the code. Take a look into the `preprocess` module before.

## Customization

The submodule is designed to be easily customizable. You can modify the `_main.py` script to customize the data loading process or add additional functionality. The `_utils.py` file contains utility functions and classes that can be modified as per your requirements.

## Notes

https://www.wormatlas.org/neurons/Individual%20Neurons/Neuronframeset.html 
The total counts of both electrical and chemical synapses are likely to be substantially higher than what was reported in the Mind of a Worm. These synapses are also being given “weights” based on how many thin sections in which each category of contact has been seen (pers pomm. Cook, Emmons, Hall et al., 2015).