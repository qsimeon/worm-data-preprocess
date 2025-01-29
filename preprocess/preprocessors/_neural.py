"""
Contains dataset-specific neural preprocessors,
which are subclasses of NeuralBasePreprocessor.

Count: 12
"""
from preprocess._pkg import *
from preprocess.preprocessors._helpers import *
from preprocess.preprocessors._base_preprocessors import NeuralBasePreprocessor


class Kato2015Preprocessor(NeuralBasePreprocessor):
    """
    Preprocessor for the Kato et al., 2015 neural data.

    This class extends the NeuralBasePreprocessor to provide specific preprocessing
    steps for the Kato et al., 2015 neural data. It includes methods for loading,
    processing, and saving the neural data.

    Methods:
        extract_data(arr): Extracts neuron IDs, calcium traces, and time vector from the loaded data array.
        preprocess(): Preprocesses the Kato et al., 2015 neural data and saves a pickle file.
    """

    def __init__(self, transform, smooth_method, interpolate_method, resample_dt, **kwargs):
        """
        Initialize the Kato2015Preprocessor with the provided parameters.

        Parameters:
            transform (object): The sklearn transformation to be applied to the data.
            smooth_method (str): The smoothing method to apply to the data.
            interpolate_method (str): The interpolation method to use when resampling the data.
            resample_dt (float): The resampling time interval in seconds.
            **kwargs: Additional keyword arguments for smoothing.
        """
        super().__init__(
            "Kato2015",
            transform,
            smooth_method,
            interpolate_method,
            resample_dt,
            **kwargs,
        )
        self.citation = "Kato, S., Kaplan, H. S., Schrödel, T., Skora, S., Lindsay, T. H., Yemini, E., Lockery, S., & Zimmer, M. (2015). Global brain dynamics embed the motor command sequence of Caenorhabditis elegans. Cell, 163(3), 656–669. https://doi.org/10.1016/j.cell.2015.09.034"

    def extract_data(self, arr):
        """
        Extracts neuron IDs, calcium traces, and time vector from the loaded data array.

        Parameters:
            arr (dict): The loaded data array.

        Returns:
            tuple: A tuple containing neuron IDs, calcium traces, and time vector.
        """
        # Identified neuron IDs (only subset have neuron names)
        all_IDs = arr["IDs"] if "IDs" in arr.keys() else arr["NeuronNames"]
        # Neural activity traces corrected for bleaching
        all_traces = arr["traces"] if "traces" in arr.keys() else arr["deltaFOverF_bc"]
        # Time vector in seconds
        timeVectorSeconds = (
            arr["timeVectorSeconds"] if "timeVectorSeconds" in arr.keys() else arr["tv"]
        )
        # Return the extracted data
        return all_IDs, all_traces, timeVectorSeconds

    def preprocess(self):
        """
        Preprocesses the Kato et al., 2015 neural data and saves it as a pickle file.

        The data is read from MAT files named "WT_Stim.mat" and "WT_NoStim.mat".

        Steps:
            1. Initialize an empty dictionary for preprocessed data and a worm index.
            2. Iterate through the MAT files:
                - Load the data from the MAT file.
                - Extract neuron IDs, calcium traces, and time vector from the loaded data.
                - Preprocess the traces and update the preprocessed data dictionary.
            3. Reshape the calcium data for each worm.
            4. Save the preprocessed data to a pickle file.
        """
        # Store preprocessed data
        preprocessed_data = dict()
        # Initialize worm index outside file loop
        worm_idx = 0
        # Load and preprocess data
        for file_name in ["WT_Stim.mat", "WT_NoStim.mat"]:
            data_key = file_name.split(".")[0]
            raw_data = self.load_data(file_name)[data_key]  # load
            neuron_IDs, traces, raw_timeVectorSeconds = self.extract_data(raw_data)  # extract
            metadata = dict(
                citation=self.citation,
                data_file=os.path.join(
                    os.path.basename(self.raw_data_path), self.source_dataset, file_name
                ),
            )
            preprocessed_data, worm_idx = self.preprocess_traces(
                neuron_IDs,
                traces,
                raw_timeVectorSeconds,
                preprocessed_data,
                worm_idx,
                metadata,
            )  # preprocess
        # Reshape calcium data
        for worm in preprocessed_data.keys():
            preprocessed_data[worm] = reshape_calcium_data(preprocessed_data[worm])
        # Save data
        self.save_data(preprocessed_data)
        # logger.info(f"Finished processing {self.source_dataset}.")
        

class Nichols2017Preprocessor(NeuralBasePreprocessor):
    """
    Preprocessor for the Nichols et al., 2017 neural data.

    This class extends the NeuralBasePreprocessor to provide specific preprocessing
    steps for the Nichols et al., 2017 neural data. It includes methods for loading,
    processing, and saving the neural data.

    Methods:
        extract_data(arr): Extracts neuron IDs, calcium traces, and time vector from the loaded data array.
        preprocess(): Preprocesses the Nichols et al., 2017 neural data and saves it as a pickle file.
    """

    def __init__(self, transform, smooth_method, interpolate_method, resample_dt, **kwargs):
        """
        Initialize the Nichols2017Preprocessor with the provided parameters.

        Parameters:
            transform (object): The sklearn transformation to be applied to the data.
            smooth_method (str): The smoothing method to apply to the data.
            interpolate_method (str): The interpolation method to use when resampling the data.
            resample_dt (float): The resampling time interval in seconds.
            **kwargs: Additional keyword arguments for smoothing.
        """
        super().__init__(
            "Nichols2017",
            transform,
            smooth_method,
            interpolate_method,
            resample_dt,
            **kwargs,
        )
        self.citation = "Nichols, A. L. A., Eichler, T., Latham, R., & Zimmer, M. (2017). A global brain state underlies C. elegans sleep behavior. Science, 356(6344). https://doi.org/10.1126/science.aam6851"

    def extract_data(self, arr):
        """
        Extracts neuron IDs, calcium traces, and time vector from the loaded data array.

        Parameters:
            arr (dict): The loaded data array.

        Returns:
            tuple: A tuple containing neuron IDs, calcium traces, and time vector.
        """
        # Identified neuron IDs (only subset have neuron names)
        all_IDs = arr["IDs"]
        # Neural activity traces corrected for bleaching
        all_traces = arr["traces"]
        # Time vector in seconds
        timeVectorSeconds = arr["timeVectorSeconds"]
        # Return the extracted data
        return all_IDs, all_traces, timeVectorSeconds

    def preprocess(self):
        """
        Preprocesses the Nichols et al., 2017 neural data and saves it as a pickle file.

        The data is read from MAT files named "n2_let.mat", "n2_prelet.mat", "npr1_let.mat", and "npr1_prelet.mat".

        Steps:
            1. Initialize an empty dictionary for preprocessed data and a worm index.
            2. Iterate through the MAT files:
                - Load the data from the MAT file.
                - Extract neuron IDs, calcium traces, and time vector from the loaded data.
                - Preprocess the traces and update the preprocessed data dictionary.
            3. Reshape the calcium data for each worm.
            4. Save the preprocessed data to a pickle file.
        """
        # Store preprocessed data
        preprocessed_data = dict()
        # Initialize worm index outside file loop
        worm_idx = 0
        # Each file contains data of many worms under the same experiment condition
        for file_name in [
            "n2_let.mat",  # let = lethargus = late larval stage 4 (L4)
            "n2_prelet.mat",  # n2 = standard lab strain, more solitary
            "npr1_let.mat",  # npr-1 = proxy for wild-type strain, more social
            "npr1_prelet.mat",  # prelet = pre-lethargus = mid-L4 stage
        ]:
            data_key = file_name.split(".")[0]
            raw_data = self.load_data(file_name)[data_key]  # load
            neuron_IDs, traces, raw_timeVectorSeconds = self.extract_data(raw_data)  # extract
            metadata = dict(
                citation=self.citation,
                data_file=os.path.join(
                    os.path.basename(self.raw_data_path), self.source_dataset, file_name
                ),
            )
            preprocessed_data, worm_idx = self.preprocess_traces(
                neuron_IDs,
                traces,
                raw_timeVectorSeconds,
                preprocessed_data,
                worm_idx,
                metadata,
            )  # preprocess
        # Reshape calcium data
        for worm in preprocessed_data.keys():
            preprocessed_data[worm] = reshape_calcium_data(preprocessed_data[worm])
        # Save data
        self.save_data(preprocessed_data)
        # logger.info(f"Finished processing {self.source_dataset}.")


class Skora2018Preprocessor(NeuralBasePreprocessor):
    """
    Preprocessor for the Skora et al., 2018 neural data.

    This class extends the NeuralBasePreprocessor to provide specific preprocessing
    steps for the Skora et al., 2018 neural data. It includes methods for loading,
    processing, and saving the neural data.

    Methods:
        extract_data(arr): Extracts neuron IDs, calcium traces, and time vector from the loaded data array.
        preprocess(): Preprocesses the Skora et al., 2018 neural data and saves it as a pickle file.
    """

    def __init__(self, transform, smooth_method, interpolate_method, resample_dt, **kwargs):
        """
        Initialize the Skora2018Preprocessor with the provided parameters.

        Parameters:
            transform (object): The sklearn transformation to be applied to the data.
            smooth_method (str): The smoothing method to apply to the data.
            interpolate_method (str): The interpolation method to use when resampling the data.
            resample_dt (float): The resampling time interval in seconds.
            **kwargs: Additional keyword arguments for smoothing.
        """
        super().__init__(
            "Skora2018",
            transform,
            smooth_method,
            interpolate_method,
            resample_dt,
            **kwargs,
        )
        self.citation = "Skora, S., Mende, F., & Zimmer, M. (2018). Energy Scarcity Promotes a Brain-wide Sleep State Modulated by Insulin Signaling in C. elegans. Cell Reports, 22(4), 953–966. https://doi.org/10.1016/j.celrep.2017.12.091"

    def extract_data(self, arr):
        """
        Extracts neuron IDs, calcium traces, and time vector from the loaded data array.

        Parameters:
            arr (dict): The loaded data array.

        Returns:
            tuple: A tuple containing neuron IDs, calcium traces, and time vector.
        """
        # Identified neuron IDs (only subset have neuron names)
        all_IDs = arr["IDs"]
        # Neural activity traces corrected for bleaching
        all_traces = arr["traces"]
        # Time vector in seconds
        timeVectorSeconds = arr["timeVectorSeconds"]
        # Return the extracted data
        return all_IDs, all_traces, timeVectorSeconds

    def preprocess(self):
        """
        Preprocesses the Skora et al., 2018 neural data and saves it as a pickle file.

        The data is read from MAT files named "WT_fasted.mat" and "WT_starved.mat".

        Steps:
            1. Initialize an empty dictionary for preprocessed data and a worm index.
            2. Iterate through the MAT files:
                - Load the data from the MAT file.
                - Extract neuron IDs, calcium traces, and time vector from the loaded data.
                - Preprocess the traces and update the preprocessed data dictionary.
            3. Reshape the calcium data for each worm.
            4. Save the preprocessed data to a pickle file.
        """
        # Store preprocessed data
        preprocessed_data = dict()
        # Initialize worm index outside file loop
        worm_idx = 0
        # Load and preprocess data
        for file_name in ["WT_fasted.mat", "WT_starved.mat"]:
            data_key = file_name.split(".")[0]
            raw_data = self.load_data(file_name)[data_key]  # load
            neuron_IDs, traces, raw_timeVectorSeconds = self.extract_data(raw_data)  # extract
            metadata = dict(
                citation=self.citation,
                data_file=os.path.join(
                    os.path.basename(self.raw_data_path), self.source_dataset, file_name
                ),
            )
            preprocessed_data, worm_idx = self.preprocess_traces(
                neuron_IDs,
                traces,
                raw_timeVectorSeconds,
                preprocessed_data,
                worm_idx,
                metadata,
            )  # preprocess
        # Reshape calcium data
        for worm in preprocessed_data.keys():
            preprocessed_data[worm] = reshape_calcium_data(preprocessed_data[worm])
        # Save data
        self.save_data(preprocessed_data)
        # logger.info(f"Finished processing {self.source_dataset}.")


class Kaplan2020Preprocessor(NeuralBasePreprocessor):
    """
    Preprocessor for the Kaplan et al., 2020 neural data.

    This class extends the NeuralBasePreprocessor to provide specific preprocessing
    steps for the Kaplan et al., 2020 neural data. It includes methods for loading,
    processing, and saving the neural data.

    Methods:
        load_data(file_name): Loads the data from the specified MAT file.
        extract_data(arr): Extracts neuron IDs, calcium traces, and time vector from the loaded data array.
        preprocess(): Preprocesses the Kaplan et al., 2020 neural data and saves a pickle file.
    """

    def __init__(self, transform, smooth_method, interpolate_method, resample_dt, **kwargs):
        """
        Initialize the Kaplan2020Preprocessor with the provided parameters.

        Parameters:
            transform (object): The sklearn transformation to be applied to the data.
            smooth_method (str): The smoothing method to apply to the data.
            interpolate_method (str): The interpolation method to use when resampling the data.
            resample_dt (float): The resampling time interval in seconds.
            **kwargs: Additional keyword arguments for smoothing.
        """
        super().__init__(
            "Kaplan2020",
            transform,
            smooth_method,
            interpolate_method,
            resample_dt,
            **kwargs,
        )
        self.citation = "Kaplan, H. S., Salazar Thula, O., Khoss, N., & Zimmer, M. (2020). Nested Neuronal Dynamics Orchestrate a Behavioral Hierarchy across Timescales. Neuron, 105(3), 562-576.e9. https://doi.org/10.1016/j.neuron.2019.10.037"

    def load_data(self, file_name):
        """
        Loads the data from the specified MAT file.

        Parameters:
            file_name (str): The name of the MAT file containing the data.

        Returns:
            dict: The loaded data as a dictionary.
        """
        # Load data with mat73
        data = mat73.loadmat(os.path.join(self.raw_data_path, self.source_dataset, file_name))
        return data

    def extract_data(self, arr):
        """
        Extracts neuron IDs, calcium traces, and time vector from the loaded data array.

        Parameters:
            arr (dict): The loaded data array.

        Returns:
            tuple: A tuple containing neuron IDs, calcium traces, and time vector.
        """
        # Identified neuron IDs (only subset have neuron names)
        all_IDs = arr["neuron_ID"]
        # Neural activity traces corrected for bleaching
        all_traces = arr["traces_bleach_corrected"]
        # Time vector in seconds
        timeVectorSeconds = arr["time_vector"]
        # Return the extracted data
        return all_IDs, all_traces, timeVectorSeconds

    def preprocess(self):
        """
        Preprocesses the Kaplan et al., 2020 neural data and saves it as a pickle file.

        The data is read from MAT files named "Neuron2019_Data_MNhisCl_RIShisCl.mat", "Neuron2019_Data_RIShisCl.mat", and "Neuron2019_Data_SMDhisCl_RIShisCl.mat".

        Steps:
            1. Initialize an empty dictionary for preprocessed data and a worm index.
            2. Iterate through the MAT files:
                - Load the data from the MAT file.
                - Extract neuron IDs, calcium traces, and time vector from the loaded data.
                - Preprocess the traces and update the preprocessed data dictionary.
            3. Reshape the calcium data for each worm.
            4. Save the preprocessed data to a pickle file.
        """
        # Store preprocessed data
        preprocessed_data = dict()
        # Initialize worm index outside file loop
        worm_idx = 0
        # Each file contains data of many worms under the same experiment condition
        for file_name in [
            "Neuron2019_Data_MNhisCl_RIShisCl.mat",  # MN = motor neuron; RIS = quiescence-promoting neuron
            "Neuron2019_Data_RIShisCl.mat",  # SMD = excitatory motor neurons targeting head and neck muscle
            "Neuron2019_Data_SMDhisCl_RIShisCl.mat",  # hisCL = histamine-gated chloride channel (inhibitory)
        ]:
            data_key = "_".join((file_name.split(".")[0].strip("Neuron2019_Data_"), "Neuron2019"))
            raw_data = self.load_data(file_name)[data_key]  # load
            neuron_IDs, traces, raw_timeVectorSeconds = self.extract_data(raw_data)  # extract
            metadata = dict(
                citation=self.citation,
                data_file=os.path.join(
                    os.path.basename(self.raw_data_path), self.source_dataset, file_name
                ),
            )
            preprocessed_data, worm_idx = self.preprocess_traces(
                neuron_IDs,
                traces,
                raw_timeVectorSeconds,
                preprocessed_data,
                worm_idx,
                metadata,
            )  # preprocess
        # Reshape calcium data
        for worm in preprocessed_data.keys():
            preprocessed_data[worm] = reshape_calcium_data(preprocessed_data[worm])
        # Save data
        self.save_data(preprocessed_data)
        # logger.info(f"Finished processing {self.source_dataset}.")


class Nejatbakhsh2020Preprocessor(NeuralBasePreprocessor):
    """
    Preprocessor for the Nejatbakhsh et al., 2020 neural data.

    This class extends the NeuralBasePreprocessor to provide specific preprocessing
    steps for the Nejatbakhsh et al., 2020 neural data. It includes methods for loading,
    processing, and saving the neural data.

    Methods:
        extract_data(file): Extracts neuron IDs, calcium traces, and time vector from the NWB file.
        preprocess(): Preprocesses the Nejatbakhsh et al., 2020 neural data and saves a pickle file.
    """

    def __init__(self, transform, smooth_method, interpolate_method, resample_dt, **kwargs):
        """
        Initialize the Nejatbakhsh2020Preprocessor with the provided parameters.

        Parameters:
            transform (object): The sklearn transformation to be applied to the data.
            smooth_method (str): The smoothing method to apply to the data.
            interpolate_method (str): The interpolation method to use when resampling the data.
            resample_dt (float): The resampling time interval in seconds.
            **kwargs: Additional keyword arguments for smoothing.
        """
        super().__init__(
            "Nejatbakhsh2020",
            transform,
            smooth_method,
            interpolate_method,
            resample_dt,
            **kwargs,
        )
        self.citation = "Nejatbakhsh, A., Varol, E., Yemini, E., Venkatachalam, V., Lin, A., Samuel, A. D. T., & Paninski, L. (2020). Extracting neural signals from semi-immobilized animals with deformable non-negative matrix factorization. In bioRxiv (p. 2020.07.07.192120). https://doi.org/10.1101/2020.07.07.192120"

    def extract_data(self, file):
        """
        Extracts neuron IDs, calcium traces, and time vector from the NWB file.

        Parameters:
            file (str): The path to the NWB file.

        Returns:
            tuple: A tuple containing neuron IDs, calcium traces, and time vector.
        """
        with NWBHDF5IO(file, "r") as io:
            read_nwbfile = io.read()
            traces = np.array(
                read_nwbfile.processing["CalciumActivity"]
                .data_interfaces["SignalRawFluor"]
                .roi_response_series["SignalCalciumImResponseSeries"]
                .data
            )
            # TODO: Impute missing NaN values.
            neuron_ids = np.array(
                read_nwbfile.processing["CalciumActivity"].data_interfaces["NeuronIDs"].labels,
                dtype=np.dtype(str),
            )
            # sampling frequency is 4 Hz
            time_vector = np.arange(0, traces.shape[0]).astype(np.dtype(float)) / 4
        # Return the extracted data
        return neuron_ids, traces, time_vector

    def preprocess(self):
        """
        Preprocesses the Nejatbakhsh et al., 2020 neural data and saves it as a pickle file

        The data is read from NWB files located in subdirectories nested in this source dataset's directory.

        Steps:
            1. Initialize an empty dictionary for preprocessed data and a worm index.
            2. Iterate through the subfolders and files in the dataset directory:
                - Extract neuron IDs, calcium traces, and time vector from each NWB file.
                - Preprocess the traces and update the preprocessed data dictionary.
            3. Reshape the calcium data for each worm.
            4. Save the preprocessed data to a pickle file.
        """
        preprocessed_data = dict()
        worm_idx = 0
        for tree in tqdm(os.listdir(os.path.join(self.raw_data_path, self.source_dataset))):
            # Skip hidden/system files like .DS_Store
            if tree.startswith("."):
                continue
            subfolder = os.path.join(self.raw_data_path, self.source_dataset, tree)
            if not os.path.isdir(subfolder):
                continue
            for file_name in os.listdir(subfolder):
                # Ignore non-NWB files
                if not file_name.endswith(".nwb"):
                    continue
                neuron_ids, traces, raw_time_vector = self.extract_data(
                    os.path.join(self.raw_data_path, self.source_dataset, subfolder, file_name)
                )
                metadata = dict(
                    citation=self.citation,
                    data_file=os.path.join(
                        os.path.basename(self.raw_data_path), self.source_dataset, file_name
                    ),
                )
                preprocessed_data, worm_idx = self.preprocess_traces(
                    [neuron_ids],
                    [traces],
                    [raw_time_vector],
                    preprocessed_data,
                    worm_idx,
                    metadata,
                )
        for worm in preprocessed_data.keys():
            preprocessed_data[worm] = reshape_calcium_data(preprocessed_data[worm])
        self.save_data(preprocessed_data)
        # logger.info(f"Finished processing {self.source_dataset}.")


class Yemini2021Preprocessor(NeuralBasePreprocessor):
    """
    Preprocessor for the Yemini et al., 2021 neural data.

    This class extends the NeuralBasePreprocessor to provide specific preprocessing
    steps for the Yemini et al., 2021 neural data. It includes methods for loading,
    processing, and saving the neural data.

    Methods:
        load_data(file_name): Loads the data from the specified MAT file.
        extract_data(raw_data): Extracts neuron IDs, calcium traces, and time vector from the loaded data array.
        preprocess(): Preprocesses the Yemini et al., 2021 neural data and saves it as a pickle file.
    """

    def __init__(self, transform, smooth_method, interpolate_method, resample_dt, **kwargs):
        """
        Initialize the Yemini2021Preprocessor with the provided parameters.

        Parameters:
            transform (object): The sklearn transformation to be applied to the data.
            smooth_method (str): The smoothing method to apply to the data.
            interpolate_method (str): The interpolation method to use when resampling the data.
            resample_dt (float): The resampling time interval in seconds.
            **kwargs: Additional keyword arguments for smoothing.
        """
        super().__init__(
            "Yemini2021",
            transform,
            smooth_method,
            interpolate_method,
            resample_dt,
            **kwargs,
        )
        self.citation = "Yemini, E., Lin, A., Nejatbakhsh, A., Varol, E., Sun, R., Mena, G. E., Samuel, A. D. T., Paninski, L., Venkatachalam, V., & Hobert, O. (2021). NeuroPAL: A Multicolor Atlas for Whole-Brain Neuronal Identification in C. elegans. Cell, 184(1), 272-288.e11. https://doi.org/10.1016/j.cell.2020.12.012"

    def load_data(self, file_name):
        """
        Loads the data from the specified MAT file.

        Parameters:
            file_name (str): The name of the MAT file containing the data.

        Returns:
            dict: The loaded data as a dictionary.
        """
        # Overriding the base class method to use scipy.io.loadmat for .mat files
        data = loadmat(os.path.join(self.raw_data_path, self.source_dataset, file_name))
        return data

    def extract_data(self, raw_data):
        """
        Extracts neuron IDs, calcium traces, and time vector from the loaded data array.

        Parameters:
            raw_data (dict): The loaded data array.

        Returns:
            tuple: A tuple containing neuron IDs, calcium traces, and time vector.

        Steps:
            1. Extract frames per second (fps) from the raw data.
            2. Extract the list of files, bilateral neurons, and boolean masks for left/right neurons.
            3. Extract histogram-normalized neuronal traces.
            4. Initialize lists for neuron IDs, traces, and time vectors.
            5. Iterate through each file in the list of files:
                - Initialize lists for neurons, activity, and time vector for the current file.
                - Iterate through each neuron in the list of bilateral neurons:
                    - Assign neuron names with L/R and get associated traces.
                    - Handle non-bilaterally symmetric neurons.
                    - Handle bilaterally symmetric neurons and assign left/right traces.
                    - Update the time vector if necessary.
                - Append the neurons, activity, and time vector for the current file to the respective lists.
            6. Return the extracted neuron IDs, traces, and time vectors.
        """
        # Frames per second
        fps = raw_data["fps"].item()
        # There several files (each is data for one worm) in each .mat file
        files = [_.item() for _ in raw_data["files"].squeeze()]
        # The list `bilat_neurons` does not disambiguate L/R neurons, so we need to do that
        bilat_neurons = [_.item() for _ in raw_data["neurons"].squeeze()]
        # List of lists. Outer list same length as `neuron`. Inner lists are boolean masks for L/R neurons organized by file in `files`.
        is_left_neuron = [  # in each inner list, all L (1) neurons appear before all R (0) neurons
            _.squeeze().tolist() for _ in raw_data["is_L"].squeeze()
        ]  # non-bilateral neurons are nan
        # Histogram-normalized neuronal traces linearly scaled and offset so that neurons are comparable
        norm_traces = [
            _.squeeze().tolist() for _ in raw_data["norm_traces"].squeeze()
        ]  # list-of-lists like `is_left_neuron`
        # This part is the meat of the `extract_data` method
        neuron_IDs = []
        traces = []
        time_vector_seconds = []
        # Each file contains data for one worm
        for f, file in enumerate(files):
            neurons = []
            activity = []
            tvec = np.empty(0, dtype=np.float32)
            for i, neuron in enumerate(bilat_neurons):
                # Assign neuron names with L/R and get associated traces
                bilat_bools = is_left_neuron[i]  # tells us if neuron is L/R
                bilat_traces = norm_traces[i]
                assert len(bilat_traces) == len(
                    bilat_bools
                ), f"Something is wrong with the data. Traces don't match with bilateral mask: {len(bilat_traces)} != {len(bilat_bools)}"
                righty = None
                if len(bilat_bools) // len(files) == 2:
                    # Get lateral assignment
                    lefty = bilat_bools[: len(bilat_bools) // 2][f]
                    righty = bilat_bools[len(bilat_bools) // 2 :][f]
                    # Get traces
                    left_traces = bilat_traces[: len(bilat_traces) // 2][f]
                    right_traces = bilat_traces[len(bilat_traces) // 2 :][f]
                elif len(bilat_bools) == len(files):
                    # Get lateral assignment
                    lefty = bilat_bools[:][f]
                    righty = None
                    # Get traces
                    left_traces = bilat_traces[:][f]
                    right_traces = None
                else:
                    raise ValueError(
                        f"Something is wrong with the data.\nNeuron: {neuron}. File: {file}."
                    )
                if np.isnan(lefty):  # non-bilaterally symmetric neuron
                    act = bilat_traces[f].squeeze().astype(float)
                    neurons.append(None if act.size == 0 else f"{neuron}")
                    activity.append(act)
                else:
                    if lefty == 1:  # left neuron
                        act = left_traces.squeeze().astype(float)
                        neurons.append(None if act.size == 0 else f"{neuron}L")
                        activity.append(act)
                    if righty != None:  # right neuron
                        act = right_traces.squeeze().astype(float)
                        tvec = np.arange(act.size) / fps
                        neurons.append(None if act.size == 0 else f"{neuron}R")
                        activity.append(act)
                # Deal with  time vector which should be the same across all neurons
                if act.size > 0 and act.size > tvec.size:
                    tvec = np.arange(act.size) / fps
            # Add neurons to list of neuron_IDs
            neuron_IDs.append(neurons)
            # Reshape activity to be a 2D array with shape (time, neurons)
            activity = np.stack(
                [
                    np.zeros_like(tvec, dtype=np.float32) if act.size == 0 else act
                    for act in activity
                ],
                dtype=np.float32,
            ).T  # (time, neurons)
            # Observed empirically that the first three values of activity equal 0.0s
            activity = activity[4:]
            tvec = tvec[4:]
            # Impute any remaining NaN values
            # NOTE: This is very slow with the default settings!
            imputer = IterativeImputer(random_state=0, n_nearest_features=10, skip_complete=False)
            if np.isnan(activity).any():
                activity = imputer.fit_transform(activity)
            # Add activity to list of traces
            traces.append(activity)
            # Add time vector to list of time vectors
            time_vector_seconds.append(tvec)
        # Return the extracted data
        return neuron_IDs, traces, time_vector_seconds

    def preprocess(self):
        """
        Preprocesses the Yemini et al., 2021 neural data and saves it as a pickle file.

        The data is read from MAT files named "Head_Activity_OH15500.mat", "Head_Activity_OH16230.mat", and "Tail_Activity_OH16230.mat".

        Steps:
            1. Initialize an empty dictionary for preprocessed data and a worm index.
            2. Iterate through the MAT files:
                - Load the data from the MAT file.
                - Extract neuron IDs, calcium traces, and time vector from the loaded data.
                - Preprocess the traces and update the preprocessed data dictionary.
            3. Reshape the calcium data for each worm.
            4. Save the preprocessed data to a pickle file.
        """
        # Store preprocessed data
        preprocessed_data = dict()
        # Initialize worm index outside file loop
        worm_idx = 0
        # Multiple .mat files to iterate over
        for file_name in [
            "Head_Activity_OH15500.mat",
            "Head_Activity_OH16230.mat",
            "Tail_Activity_OH16230.mat",
        ]:
            raw_data = self.load_data(file_name)  # load
            neuron_IDs, traces, raw_timeVectorSeconds = self.extract_data(raw_data)  # extract
            metadata = dict(
                citation=self.citation,
                data_file=os.path.join(
                    os.path.basename(self.raw_data_path), self.source_dataset, file_name
                ),
            )
            preprocessed_data, worm_idx = self.preprocess_traces(
                neuron_IDs,
                traces,
                raw_timeVectorSeconds,
                preprocessed_data,
                worm_idx,
                metadata,
            )  # preprocess
        # Reshape calcium data
        for worm in preprocessed_data.keys():
            preprocessed_data[worm] = reshape_calcium_data(preprocessed_data[worm])
        # Save data
        self.save_data(preprocessed_data)
        # logger.info(f"Finished processing {self.source_dataset}.")


class Uzel2022Preprocessor(NeuralBasePreprocessor):
    """
    Preprocessor for the Uzel et al., 2022 connectome data.

    This class extends the NeuralBasePreprocessor to provide specific preprocessing
    steps for the Uzel et al., 2022 connectome data. It includes methods for loading,
    processing, and saving the connectome data.

    Methods:
        load_data(file_name): Loads the data from the specified MAT file.
        extract_data(arr): Extracts neuron IDs, calcium traces, and time vector from the loaded data array.
        preprocess(): Preprocesses the Uzel et al., 2022 neural data and saves is as a file.
    """

    def __init__(self, transform, smooth_method, interpolate_method, resample_dt, **kwargs):
        """
        Initialize the Uzel2022Preprocessor with the provided parameters.

        Parameters:
            transform (object): The sklearn transformation to be applied to the data.
            smooth_method (str): The smoothing method to apply to the data.
            interpolate_method (str): The interpolation method to use when resampling the data.
            resample_dt (float): The resampling time interval in seconds.
            **kwargs: Additional keyword arguments for smoothing.
        """
        super().__init__(
            "Uzel2022",
            transform,
            smooth_method,
            interpolate_method,
            resample_dt,
            **kwargs,
        )
        self.citation = "Uzel, K., Kato, S., & Zimmer, M. (2022). A set of hub neurons and non-local connectivity features support global brain dynamics in C. elegans. Current Biology: CB, 32(16), 3443-3459.e8. https://doi.org/10.1016/j.cub.2022.06.039"

    def load_data(self, file_name):
        """
        Loads the data from the specified MAT file.

        Parameters:
            file_name (str): The name of the MAT file containing the data.

        Returns:
            dict: The loaded data as a dictionary.
        """
        # Load data with mat73
        return mat73.loadmat(os.path.join(self.raw_data_path, self.source_dataset, file_name))

    def extract_data(self, arr):
        """
        Extracts neuron IDs, calcium traces, and time vector from the loaded data array.

        Parameters:
            arr (dict): The loaded data array.

        Returns:
            tuple: A tuple containing neuron IDs, calcium traces, and time vector.
        """
        # Identified neuron IDs (only subset have neuron names)
        all_IDs = arr["IDs"]
        # Neural activity traces corrected for bleaching
        all_traces = arr["traces"]  # (time, neurons)
        # Time vector in seconds
        timeVectorSeconds = arr["tv"]
        # Return the extracted data
        return all_IDs, all_traces, timeVectorSeconds

    def preprocess(self):
        """
        Preprocesses the Uzel et al., 2022 neural data and saves it as a pickle file.

        The data is read from a MAT file named "Uzel_WT.mat".

        Parameters:
            save_as (str, optional): The name of the file to save the graph tensors to. Default is "graph_tensors_uzel2022.pt".

        Steps:
            1. Initialize an empty dictionary for preprocessed data and a worm index.
            2. Load the data from the MAT file.
            3. Extract neuron IDs, calcium traces, and time vector from the loaded data.
            4. Preprocess the traces and update the preprocessed data dictionary.
            5. Reshape the calcium data for each worm.
            6. Save the preprocessed data to the specified file.
        """
        preprocessed_data = dict()  # Store preprocessed data
        worm_idx = 0  # Initialize worm index outside file loop
        # Load and preprocess data
        for file_name in ["Uzel_WT.mat"]:
            data_key = "Uzel_WT"
            raw_data = self.load_data(file_name)[data_key]  # load
            neuron_IDs, traces, raw_timeVectorSeconds = self.extract_data(raw_data)  # extract
            metadata = dict(
                citation=self.citation,
                data_file=os.path.join(
                    os.path.basename(self.raw_data_path), self.source_dataset, file_name
                ),
            )
            preprocessed_data, worm_idx = self.preprocess_traces(
                neuron_IDs,
                traces,
                raw_timeVectorSeconds,
                preprocessed_data,
                worm_idx,
                metadata,
            )  # preprocess
        # Reshape calcium data
        for worm in preprocessed_data.keys():
            preprocessed_data[worm] = reshape_calcium_data(preprocessed_data[worm])
        # Save data
        self.save_data(preprocessed_data)
        # logger.info(f"Finished processing {self.source_dataset}.")


class Dag2023Preprocessor(NeuralBasePreprocessor):
    """
    Preprocessor for the Dag et al., 2023 connectome data.

    This class extends the NeuralBasePreprocessor to provide specific preprocessing
    steps for the Dag et al., 2023 connectome data. It includes methods for loading,
    processing, and saving the connectome data.

    Methods:
        load_data(file_name): Loads the data from the specified HDF5 file.
        load_labels_dict(labels_file="NeuroPAL_labels_dict.json"): Loads the neuron labels dictionary from a JSON file.
        find_nearest_label(query, possible_labels, char="?"): Finds the nearest neuron label from a list given a query.
        extract_data(data_file, labels_file): Extracts neuron IDs, calcium traces, and time vector from the loaded data file.
        preprocess(): Preprocesses the Dag et al., 2023 neural data and saves it as a pickle file.
    """

    def __init__(self, transform, smooth_method, interpolate_method, resample_dt, **kwargs):
        """
        Initialize the Dag2023Preprocessor with the provided parameters.

        Parameters:
            transform (object): The sklearn transformation to be applied to the data.
            smooth_method (str): The smoothing method to apply to the data.
            interpolate_method (str): The interpolation method to use when resampling the data.
            resample_dt (float): The resampling time interval in seconds.
            **kwargs: Additional keyword arguments for smoothing.
        """
        super().__init__(
            "Dag2023",
            transform,
            smooth_method,
            interpolate_method,
            resample_dt,
            **kwargs,
        )
        self.citation = "Dag, U., Nwabudike, I., Kang, D., Gomes, M. A., Kim, J., Atanas, A. A., Bueno, E., Estrem, C., Pugliese, S., Wang, Z., Towlson, E., & Flavell, S. W. (2023). Dissecting the functional organization of the C. elegans serotonergic system at whole-brain scale. Cell, 186(12), 2574-2592.e20. https://doi.org/10.1016/j.cell.2023.04.023"

    def load_data(self, file_name):
        """
        Loads the data from the specified HDF5 file.

        Parameters:
            file_name (str): The name of the HDF5 file containing the data.

        Returns:
            h5py.File: The loaded data as an HDF5 file object.
        """
        data = h5py.File(os.path.join(self.raw_data_path, self.source_dataset, file_name), "r")
        return data

    def load_labels_dict(self, labels_file="NeuroPAL_labels_dict.json"):
        """
        Loads the neuron labels dictionary from a JSON file.

        Parameters:
            labels_file (str, optional): The name of the JSON file containing the neuron labels. Default is "NeuroPAL_labels_dict.json".

        Returns:
            dict: The loaded neuron labels dictionary.
        """
        with open(os.path.join(self.raw_data_path, self.source_dataset, labels_file), "r") as f:
            label_info = json.load(f)
        return label_info

    def extract_data(self, data_file, labels_file):
        """
        Extracts neuron IDs, calcium traces, and time vector from the loaded data file.

        Parameters:
            data_file (str): The path to the HDF5 data file.
            labels_file (str): The path to the JSON file containing neuron labels.

        Returns:
            tuple: A tuple containing neuron IDs, calcium traces, and time vector.

        Steps:
            1. Load the data file and labels file.
            2. Extract the mapping of indices in the data to neuron labels.
            3. Extract neural activity traces and time vector.
            4. Get neuron labels corresponding to indices in calcium data.
            5. Handle ambiguous neuron labels.
            6. Return the extracted data.
        """
        # Load the data file and labels file
        file_data = self.load_data(data_file)
        label_info = self.load_labels_dict(labels_file)
        # Extract the mapping of indices in the data to neuron labels
        index_map, _ = label_info.get(data_file.split("/")[-1].strip("-data.h5"), (dict(), None))
        # Neural activity traces
        calcium = np.array(file_data["gcamp"]["traces_array_F_F20"])  # (time, neurons)
        # Time vector in seconds
        timevec = np.array(file_data["timing"]["timestamp_confocal"])[: calcium.shape[0]]  # (time,)
        # Get neuron labels corresponding to indices in calcium data
        indices = []
        neurons = []
        # If there is an index map, use it to extract the labeled neurons
        if index_map:
            # Indices in index_map correspond to labeled neurons
            for calnum in index_map:
                # NOTE: calnum is a string, not an integer
                assert (
                    int(calnum) <= calcium.shape[1]
                ), f"Index out of range. calnum: {calnum}, calcium.shape[1]: {calcium.shape[1]}"
                lbl = index_map[calnum]["label"]
                neurons.append(lbl)
                # Need to minus one because Julia index starts at 1 whereas Python index starts with 0
                idx = int(calnum) - 1
                indices.append(idx)
            # Remaining indices correspond to unlabeled neurons
            for i in range(calcium.shape[1]):
                if i not in set(indices):
                    indices.append(i)
                    neurons.append(str(i))
        # Otherwise, use the indices as the neuron labels for all traces
        else:
            indices = list(range(calcium.shape[1]))
            neurons = [str(i) for i in indices]
        # Ensure only calcium data at selected indices is kept
        calcium = calcium[:, indices]
        # Neurons with DV/LR ambiguity have '?' or '??' in labels that must be inferred
        neurons_copy = []
        for label in neurons:
            # If the neuron is unknown it will have a numeric label corresponding to its index
            if label.isnumeric():
                neurons_copy.append(label)
                continue
            # Look for the closest neuron label that will match the current string containing '?'
            replacement, _ = self.find_nearest_label(
                label, set(NEURON_LABELS) - set(neurons_copy), char="?"
            )
            neurons_copy.append(replacement)
        # Make the extracted data into a list of arrays
        all_IDs = [neurons_copy]
        all_traces = [calcium]
        timeVectorSeconds = [timevec]
        # Return the extracted data
        return all_IDs, all_traces, timeVectorSeconds

    def preprocess(self):
        """
        Preprocesses the Dag et al., 2023 neural data and saves it as a pickle file.

        The data is read from HDF5 files located in the dataset directory.

        Steps:
            1. Initialize an empty dictionary for preprocessed data and a worm index.
            2. Iterate through the subfolders and files in the dataset directory:
                - Extract neuron IDs, calcium traces, and time vector from each HDF5 file.
                - Preprocess the traces and update the preprocessed data dictionary.
            3. Reshape the calcium data for each worm.
            4. Save the preprocessed data to a pickle file.
        """
        # Load and preprocess data
        preprocessed_data = dict()  # Store preprocessed data
        worm_idx = 0  # Initialize worm index outside file loop
        # There are two subfolders in the Dag2023 dataset: 'swf415_no_id' and 'swf702_with_id'
        withid_data_files = os.path.join(self.raw_data_path, "Dag2023", "swf702_with_id")
        noid_data_files = os.path.join(self.raw_data_path, "Dag2023", "swf415_no_id")  # unused
        # 'NeuroPAL_labels_dict.json' maps data file names to a dictionary of neuron label information
        labels_file = "NeuroPAL_labels_dict.json"
        # First deal with the swf702_with_id which contains data from labeled neurons
        for file_name in os.listdir(withid_data_files):
            if not file_name.endswith(".h5"):
                continue
            data_file = os.path.join("swf702_with_id", file_name)
            neurons, raw_traces, time_vector_seconds = self.extract_data(data_file, labels_file)
            metadata = dict(
                citation=self.citation,
                data_file=os.path.join(
                    os.path.basename(self.raw_data_path), self.source_dataset, file_name
                ),
            )
            preprocessed_data, worm_idx = self.preprocess_traces(
                neurons,
                raw_traces,
                time_vector_seconds,
                preprocessed_data,
                worm_idx,
                metadata,
            )  # preprocess
        # Next deal with the swf415_no_id which contains purely unlabeled neuron data
        # NOTE: These won't get used at all as they are skipped in NeuralBasePreprocessor.preprocess_traces since num_labeled_neurons is 0.
        for file_name in os.listdir(noid_data_files):
            if not file_name.endswith(".h5"):
                continue
            data_file = os.path.join("swf415_no_id", file_name)
            neurons, raw_traces, time_vector_seconds = self.extract_data(data_file, labels_file)
            metadata = dict(
                citation=self.citation,
                data_file=os.path.join(
                    os.path.basename(self.raw_data_path), self.source_dataset, file_name
                ),
            )
            preprocessed_data, worm_idx = self.preprocess_traces(
                neurons,
                raw_traces,
                time_vector_seconds,
                preprocessed_data,
                worm_idx,
                metadata,
            )  # preprocess
        # Reshape calcium data
        for worm in preprocessed_data.keys():
            preprocessed_data[worm] = reshape_calcium_data(preprocessed_data[worm])
        # Save data
        self.save_data(preprocessed_data)
        # logger.info(f"Finished processing {self.source_dataset}.")
        return None


class Flavell2023Preprocessor(NeuralBasePreprocessor):
    """
    Preprocessor for the Atanas et al., 2023 neural data.

    This class extends the NeuralBasePreprocessor to provide specific preprocessing
    steps for the Atanas et al., 2023 neural data. It includes methods for loading,
    processing, and saving the neural data.

    Methods:
        find_nearest_label(query, possible_labels, char="?"): Finds the nearest neuron label from a list given a query.
        load_data(file_name): Loads the data from the specified HDF5 or JSON file.
        extract_data(file_data): Extracts neuron IDs, calcium traces, and time vector from the loaded data file.
        preprocess(): Preprocesses the Flavell et al., 2023 neural data and saves it as pickle file.
    """

    def __init__(self, transform, smooth_method, interpolate_method, resample_dt, **kwargs):
        """
        Initialize the Flavell2023Preprocessor with the provided parameters.

        Parameters:
            transform (object): The sklearn transformation to be applied to the data.
            smooth_method (str): The smoothing method to apply to the data.
            interpolate_method (str): The interpolation method to use when resampling the data.
            resample_dt (float): The resampling time interval in seconds.
            **kwargs: Additional keyword arguments for smoothing.
        """
        super().__init__(
            "Flavell2023",
            transform,
            smooth_method,
            interpolate_method,
            resample_dt,
            **kwargs,
        )
        self.citation = "Atanas, A. A., Kim, J., Wang, Z., Bueno, E., Becker, M., Kang, D., Park, J., Kramer, T. S., Wan, F. K., Baskoylu, S., Dag, U., Kalogeropoulou, E., Gomes, M. A., Estrem, C., Cohen, N., Mansinghka, V. K., & Flavell, S. W. (2023). Brain-wide representations of behavior spanning multiple timescales and states in C. elegans. Cell. https://doi.org/10.1016/j.cell.2023.07.035"

    def load_data(self, file_name):
        """
        Loads the data from the specified HDF5 or JSON file.

        Parameters:
            file_name (str): The name of the HDF5 or JSON file containing the data.

        Returns:
            dict or h5py.File: The loaded data as a dictionary or HDF5 file object.
        """
        if file_name.endswith(".h5"):
            data = h5py.File(os.path.join(self.raw_data_path, self.source_dataset, file_name), "r")
        elif file_name.endswith(".json"):
            with open(os.path.join(self.raw_data_path, self.source_dataset, file_name), "r") as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {file_name}")
        return data

    def extract_data(self, file_data):
        """
        Extracts neuron IDs, calcium traces, and time vector from the loaded data file.

        Parameters:
            file_data (dict or h5py.File): The loaded data file.

        Returns:
            tuple: A tuple containing neuron IDs, calcium traces, and time vector.

        Steps:
            1. Ensure the file data is in the expected format (dict or h5py.File).
            2. Extract the time vector in seconds.
            3. Extract raw traces and initialize the calcium data array.
            4. Extract neuron labels and handle ambiguous neuron labels.
            5. Filter for unique neuron labels and get data for unique neurons.
            6. Return the extracted data.
        """
        # The files are expected to use a JSON or H5 format
        assert isinstance(file_data, (dict, h5py.File)), f"Unsupported data type: {type(file_data)}"
        # Time vector in seconds
        time_in_seconds = np.array(file_data["timestamp_confocal"], dtype=np.float32)
        time_in_seconds = time_in_seconds.reshape((-1, 1))
        # Raw traces (list)
        raw_traces = file_data["trace_array"]
        # Max time steps (int)
        max_t = len(raw_traces[0])
        # Number of neurons (int)
        number_neurons = len(raw_traces)
        # Labels (list)
        ids = file_data["labeled"]
        # All traces
        calcium_data = np.zeros((max_t, number_neurons), dtype=np.float32)
        for i, trace in enumerate(raw_traces):
            calcium_data[:, i] = trace
        neurons = [str(i) for i in range(number_neurons)]
        for i in ids.keys():
            label = ids[str(i)]["label"]
            neurons[int(i) - 1] = label
        # Neurons with DV/LR ambiguity have '?' or '??' in labels that must be inferred
        neurons_copy = []
        for label in neurons:
            # If the neuron is unknown it will have a numeric label corresponding to its index
            if label.isnumeric():
                neurons_copy.append(label)
                continue
            # Look for the closest neuron label that will match the current string containing '?'
            replacement, _ = self.find_nearest_label(
                label, set(NEURON_LABELS) - set(neurons_copy), char="?"
            )
            neurons_copy.append(replacement)
        # Make the extracted data into a list of arrays
        all_IDs = [neurons_copy]
        all_traces = [calcium_data]
        timeVectorSeconds = [time_in_seconds]
        # Return the extracted data
        return all_IDs, all_traces, timeVectorSeconds

    def preprocess(self):
        """
        Preprocesses the Atanas et al., 2023 neural data and saves it as a pickle file.

        The data is read from HDF5 or JSON files located in the dataset directory.

        Parameters:
            save_as (str, optional): The name of the file to save the graph tensors to. Default is "graph_tensors_flavell2023.pt".

        Steps:
            1. Initialize an empty dictionary for preprocessed data.
            2. Iterate through the files in the dataset directory:
                - Load the data from the file.
                - Extract neuron IDs, calcium traces, and time vector from the loaded data.
                - Preprocess the traces and update the preprocessed data dictionary.
            3. Reshape the calcium data for each worm.
            4. Save the preprocessed data to a pickle file.
        """
        # Load and preprocess data
        preprocessed_data = dict()
        worm_idx = 0
        for file_name in os.listdir(os.path.join(self.raw_data_path, self.source_dataset)):
            if not (file_name.endswith(".h5") or file_name.endswith(".json")):
                continue
            file_data = self.load_data(file_name)  # load
            neurons, calcium_data, time_in_seconds = self.extract_data(file_data)  # extract
            metadata = dict(
                citation=self.citation,
                data_file=os.path.join(
                    os.path.basename(self.raw_data_path), self.source_dataset, file_name
                ),
            )
            preprocessed_data, worm_idx = self.preprocess_traces(
                neurons,
                calcium_data,
                time_in_seconds,
                preprocessed_data,
                worm_idx,
                metadata,
            )  # preprocess
        # Reshape calcium data
        for worm in preprocessed_data.keys():
            preprocessed_data[worm] = reshape_calcium_data(preprocessed_data[worm])
        # Save data
        self.save_data(preprocessed_data)
        # logger.info(f"Finished processing {self.source_dataset}.")


class Leifer2023Preprocessor(NeuralBasePreprocessor):
    """
    Preprocessor for the Randi et al., 2023 neural data.

    This class extends the NeuralBasePreprocessor to provide specific preprocessing
    steps for the Leifer et al., 2023 neural data. It includes methods for loading,
    processing, and saving the neural data.

    Methods:
        str_to_float(str_num): Converts a string in scientific notation to a floating-point number.
        load_labels(file_path): Loads neuron labels from a text file.
        load_time_vector(file_path): Loads the time vector from a text file.
        load_data(file_path): Loads the neural data from a text file.
        create_neuron_idx(label_list): Creates a mapping of neuron labels to indices.
        extract_data(data_file, labels_file, time_file): Extracts neuron IDs, calcium traces, and time vector from the loaded data files.
        preprocess(): Preprocesses the Leifer et al., 2023 neural data and saves it as a pickle a file.
    """

    def __init__(self, transform, smooth_method, interpolate_method, resample_dt, **kwargs):
        """
        Initialize the Leifer2023Preprocessor with the provided parameters.

        Parameters:
            transform (object): The sklearn transformation to be applied to the data.
            smooth_method (str): The smoothing method to apply to the data.
            interpolate_method (str): The interpolation method to use when resampling the data.
            resample_dt (float): The resampling time interval in seconds.
            **kwargs: Additional keyword arguments for smoothing.
        """
        super().__init__(
            "Leifer2023",
            transform,
            smooth_method,
            interpolate_method,
            resample_dt,
            **kwargs,
        )
        self.citation = "Randi, F., Sharma, A. K., Dvali, S., & Leifer, A. M. (2023). Neural signal propagation atlas of Caenorhabditis elegans. Nature, 623(7986), 406–414. https://doi.org/10.1038/s41586-023-06683-4"

    def str_to_float(self, str_num):
        """
        Converts a string in scientific notation to a floating-point number.

        Parameters:
            str_num (str): The string in scientific notation.

        Returns:
            float: The converted floating-point number.
        """
        before_e = float(str_num.split("e")[0])
        sign = str_num.split("e")[1][:1]
        after_e = int(str_num.split("e")[1][1:])
        if sign == "+":
            float_num = before_e * math.pow(10, after_e)
        elif sign == "-":
            float_num = before_e * math.pow(10, -after_e)
        else:
            float_num = None
            raise TypeError("Float has unknown sign.")
        return float_num

    def load_labels(self, file_path):
        """
        Loads neuron labels from a text file.

        Parameters:
            file_path (str): The path to the text file containing neuron labels.

        Returns:
            list: A list of neuron labels.
        """
        with open(file_path, "r") as f:
            labels = [line.strip("\n") for line in f.readlines()]
        return labels

    def load_time_vector(self, file_path):
        """
        Loads the time vector from a text file.

        Parameters:
            file_path (str): The path to the text file containing the time vector.

        Returns:
            np.ndarray: The time vector as a numpy array.
        """
        with open(file_path, "r") as f:
            timeVectorSeconds = [self.str_to_float(line.strip("\n")) for line in f.readlines()]
            timeVectorSeconds = np.array(timeVectorSeconds, dtype=np.float32).reshape(-1, 1)
        return timeVectorSeconds

    def load_data(self, file_path):
        """
        Loads the neural data from a text file.

        Parameters:
            file_path (str): The path to the text file containing the neural data.

        Returns:
            np.ndarray: The neural data as a numpy array.
        """
        with open(file_path, "r") as f:
            data = [list(map(float, line.split(" "))) for line in f.readlines()]
        data_array = np.array(data, dtype=np.float32)
        return data_array

    def create_neuron_idx(self, label_list):
        """
        Creates a mapping of neuron labels to indices.

        Parameters:
            label_list (list): The list of neuron labels.

        Returns:
            tuple: A tuple containing the neuron-to-index mapping and the number of labeled neurons.
        """
        neuron_to_idx = dict()
        num_unlabeled_neurons = 0
        for j, item in enumerate(label_list):
            previous_list = label_list[:j]
            if not item.isalnum():  # happens when the label is empty string ''
                label_list[j] = str(j)
                num_unlabeled_neurons += 1
                neuron_to_idx[str(j)] = j
            else:
                if item in NEURON_LABELS and item not in previous_list:
                    neuron_to_idx[item] = j
                # If a neuron label repeated assume a mistake and treat the duplicate as an unlabeled neuron
                elif item in NEURON_LABELS and item in previous_list:
                    label_list[j] = str(j)
                    num_unlabeled_neurons += 1
                    neuron_to_idx[str(j)] = j
                # Handle ambiguous neuron labels
                else:
                    if str(item + "L") in NEURON_LABELS and str(item + "L") not in previous_list:
                        label_list[j] = str(item + "L")
                        neuron_to_idx[str(item + "L")] = j
                    elif str(item + "R") in NEURON_LABELS and str(item + "R") not in previous_list:
                        label_list[j] = str(item + "R")
                        neuron_to_idx[str(item + "R")] = j
                    else:  # happens when the label is "merge"; TODO: Ask authors what that is?
                        label_list[j] = str(j)
                        num_unlabeled_neurons += 1
                        neuron_to_idx[str(j)] = j
        num_labeled_neurons = len(
            [k for k in neuron_to_idx.keys() if not k.isnumeric()]
        )  # number of labeled neurons
        assert (
            num_labeled_neurons == len(label_list) - num_unlabeled_neurons
        ), "Incorrect calculation of the number of labeled neurons."
        return neuron_to_idx, num_labeled_neurons

    def extract_data(self, data_file, labels_file, time_file):
        """
        Extracts neuron IDs, calcium traces, and time vector from the loaded data files.

        Parameters:
            data_file (str): The path to the text file containing the neural data.
            labels_file (str): The path to the text file containing neuron labels.
            time_file (str): The path to the text file containing the time vector.

        Returns:
            tuple: A tuple containing neuron IDs, calcium traces, and time vector.

        Steps:
            1. Load the neuron labels, time vector, and neural data.
            2. Filter out bad traces based on linear segments.
            3. Return the extracted data.
        """
        real_data = self.load_data(data_file)  # shaped (time, neurons)
        # In some strange cases there are more labels than neurons
        label_list = self.load_labels(labels_file)[: real_data.shape[1]]
        time_in_seconds = self.load_time_vector(time_file)
        # Check that the data, labels and time shapes match
        assert real_data.shape[1] == len(
            label_list
        ), f"Data and labels do not match!\n Files: {data_file}, {labels_file}"
        assert (
            real_data.shape[0] == time_in_seconds.shape[0]
        ), f"Time vector does not match data!\n Files: {data_file}, {time_file}"
        # Remove neuron traces that are all NaN values
        mask = np.argwhere(~np.isnan(real_data).all(axis=0)).flatten()
        real_data = real_data[:, mask]
        label_list = np.array(label_list, dtype=str)[mask].tolist()
        # Remove neurons with long stretches of NaNs
        real_data, nan_mask = self.filter_bad_traces_by_nan_stretch(real_data)
        label_list = np.array(label_list, dtype=str)[nan_mask].tolist()
        # Impute any remaining NaN values
        # NOTE: This is very slow with the default settings!
        imputer = IterativeImputer(random_state=0, n_nearest_features=10, skip_complete=False)
        if np.isnan(real_data).any():
            real_data = imputer.fit_transform(real_data)
        # Remove badly imputed neurons from the data
        filt_real_data, filt_mask = self.filter_bad_traces_by_linear_segments(real_data)
        filt_label_list = np.array(label_list, dtype=str)[filt_mask].tolist()
        # Make the extracted data into a list of arrays
        all_IDs = [filt_label_list]
        all_traces = [filt_real_data]
        timeVectorSeconds = [time_in_seconds]
        # Return the extracted data
        return all_IDs, all_traces, timeVectorSeconds

    def preprocess(self):
        """
        Preprocesses the Randi et al., 2023 neural data and saves it as a pickle file.

        The data is read from text files located in the dataset directory.

        NOTE: The `preprocess` method for the Leifer 2023 dataset is significantly different
            than that for the other datasets due to differences between the file structure containing
            the raw data for the Leifer2023 dataset compared to the other source datasets:
                - Leifer2023 raw data uses 6 files per worm each containing distinct information.
                - The other datasets use 1 file containing all the information for multiple worms.
            Unlike the `preprocess` method in the other dataset classes which makes use of the
            `preprocess_traces` method from the parent NeuralBasePreprocessor class, this one does not.

        Parameters:
            save_as (str, optional): The name of the file to save the graph tensors to. Default is "graph_tensors_leifer2023.pt".

        Steps:
            1. Initialize an empty dictionary for preprocessed data.
            2. Iterate through the files in the dataset directory:
                - Load the data from the files.
                - Extract neuron IDs, calcium traces, and time vector from the loaded data.
                - Preprocess the traces and update the preprocessed data dictionary.
            3. Reshape the calcium data for each worm.
            4. Save the preprocessed data to the specified file.
        """
        # Load and preprocess data
        preprocessed_data = dict()
        data_dir = os.path.join(self.raw_data_path, self.source_dataset)
        # Every worm has 6 text files
        files = os.listdir(data_dir)
        num_worms = int(len(files) / 6)
        # Initialize worm index outside file loop
        worm_idx = 0
        # Iterate over each worm's triad of data text files
        for i in tqdm(range(0, num_worms)):
            data_file = os.path.join(data_dir, f"{str(i)}_gcamp.txt")
            labels_file = os.path.join(data_dir, f"{str(i)}_labels.txt")
            time_file = os.path.join(data_dir, f"{str(i)}_t.txt")
            # Load and extract raw data
            label_list, real_data, time_in_seconds = self.extract_data(
                data_file, labels_file, time_file
            )  # extract
            file_name = str(i) + "_{gcamp|labels|t}.txt"
            metadata = dict(
                citation=self.citation,
                data_file=os.path.join(
                    os.path.basename(self.raw_data_path), self.source_dataset, file_name
                ),
            )
            # Preprocess raw data
            preprocessed_data, worm_idx = self.preprocess_traces(
                label_list,
                real_data,
                time_in_seconds,
                preprocessed_data,
                worm_idx,
                metadata,
            )  # preprocess
        # Reshape calcium data
        for worm in preprocessed_data.keys():
            preprocessed_data[worm] = reshape_calcium_data(preprocessed_data[worm])
        # Save data
        self.save_data(preprocessed_data)
        # logger.info(f"Finished processing {self.source_dataset}.")


class Lin2023Preprocessor(NeuralBasePreprocessor):
    """
    Preprocessor for the Lin et al., 2023 neural data.

    This class extends the NeuralBasePreprocessor to provide specific preprocessing
    steps for the Lin et al., 2023 neural data. It includes methods for loading,
    processing, and saving the neural data.

    Methods:
        load_data(file_name): Loads the data from the specified MAT file.
        extract_data(file_data): Extracts neuron IDs, calcium traces, and time vector from the loaded data file.
        create_metadata(): Creates a dictionary of extra information or metadata for the dataset.
        preprocess(): Preprocesses the Lin et al., 2023 neural data and saves it as a pickle file.
    """

    def __init__(self, transform, smooth_method, interpolate_method, resample_dt, **kwargs):
        """
        Initialize the Lin2023Preprocessor with the provided parameters.

        Parameters:
            transform (object): The sklearn transformation to be applied to the data.
            smooth_method (str): The smoothing method to apply to the data.
            interpolate_method (str): The interpolation method to use when resampling the data.
            resample_dt (float): The resampling time interval in seconds.
            **kwargs: Additional keyword arguments for smoothing.
        """
        super().__init__(
            "Lin2023",
            transform,
            smooth_method,
            interpolate_method,
            resample_dt,
            **kwargs,
        )
        self.citation = "Lin, A., Qin, S., Casademunt, H., Wu, M., Hung, W., Cain, G., Tan, N. Z., Valenzuela, R., Lesanpezeshki, L., Venkatachalam, V., Pehlevan, C., Zhen, M., & Samuel, A. D. T. (2023). Functional imaging and quantification of multineuronal olfactory responses in C. elegans. Science Advances, 9(9), eade1249. https://doi.org/10.1126/sciadv.ade1249"

    def load_data(self, file_name):
        """
        Loads the data from the specified MAT file.

        Parameters:
            file_name (str): The name of the MAT file containing the data.

        Returns:
            dict: The loaded data as a dictionary.
        """
        # Overriding the base class method to use scipy.io.loadmat for .mat files
        data = loadmat(os.path.join(self.raw_data_path, self.source_dataset, file_name))
        return data

    def extract_data(self, data_file):
        """
        Extracts neuron IDs, calcium traces, and time vector from the loaded data file.

        Parameters:
            data_file (dict): The loaded data file.

        Returns:
            tuple: A tuple containing neuron IDs, calcium traces, and time vector.

        Steps:
            1. Ensure the file data is in the expected format (dict).
            2. Extract the time vector in seconds.
            3. Extract raw traces and initialize the calcium data array.
            4. Extract neuron labels and handle ambiguous neuron labels.
            5. Filter for unique neuron labels and get data for unique neurons.
            6. Return the extracted data.
        """
        dataset_raw = self.load_data(data_file)
        # Filter for proofread neurons
        _filter = dataset_raw["use_flag"].flatten() > 0
        neurons = [str(_.item()) for _ in dataset_raw["proofread_neurons"].flatten()[_filter]]
        raw_time_vec = np.array(dataset_raw["times"].flatten()[0][-1])
        raw_activitiy = dataset_raw["corrected_F"][_filter].T  # (time, neurons)
        # Replace first nan with F0 value
        _f0 = dataset_raw["F_0"][_filter][:, 0]
        raw_activitiy[0, :] = _f0
        # Impute any remaining NaN values
        # NOTE: This is very slow with the default settings!
        imputer = IterativeImputer(random_state=0, n_nearest_features=10, skip_complete=False)
        if np.isnan(raw_activitiy).any():
            raw_activitiy = imputer.fit_transform(raw_activitiy)
        # Make the extracted data into a list of arrays
        neuron_IDs, raw_traces, time_vector_seconds = [neurons], [raw_activitiy], [raw_time_vec]
        # Return the extracted data
        return neuron_IDs, raw_traces, time_vector_seconds

    def preprocess(self):
        """
        Preprocesses the Lin et al., 2023 neural data and saves it as a pickle file.

        The data is read from MAT files located in the dataset directory.

        Parameters:
            save_as (str, optional): The name of the file to save the graph tensors to. Default is "graph_tensors_lin2023.pt".

        Steps:
            1. Initialize an empty dictionary for preprocessed data.
            2. Iterate through the files in the dataset directory:
                - Load the data from the file.
                - Extract neuron IDs, calcium traces, and time vector from the loaded data.
                - Preprocess the traces and update the preprocessed data dictionary.
            3. Reshape the calcium data for each worm.
            4. Save the preprocessed data to the specified file.
        """
        # Load and preprocess data
        preprocessed_data = dict()  # Store preprocessed data
        worm_idx = 0  # Initialize worm index outside file loop
        # Have multiple .mat files that you iterate over
        data_files = os.path.join(self.raw_data_path, "Lin2023")
        # Multiple .mat files to iterate over
        for file_name in tqdm(os.listdir(data_files)):
            if not file_name.endswith(".mat"):
                continue
            neurons, raw_traces, time_vector_seconds = self.extract_data(file_name)
            metadata = dict(
                citation=self.citation,
                data_file=os.path.join(
                    os.path.basename(self.raw_data_path), self.source_dataset, file_name
                ),
            )
            preprocessed_data, worm_idx = self.preprocess_traces(
                neurons,
                raw_traces,
                time_vector_seconds,
                preprocessed_data,
                worm_idx,
                metadata,
            )  # preprocess
        # Reshape calcium data
        for worm in preprocessed_data.keys():
            preprocessed_data[worm] = reshape_calcium_data(preprocessed_data[worm])
        # Save data
        self.save_data(preprocessed_data)
        # logger.info(f"Finished processing {self.source_dataset}.")


class Venkatachalam2024Preprocessor(NeuralBasePreprocessor):
    """
    Preprocessor for the Venkatachalam (2024, unpublished) connectome data.

    This class extends the NeuralBasePreprocessor to provide specific preprocessing
    steps for the Venkatachalam (2024, unpublished) neural data. It includes methods for loading,
    processing, and saving the neural data.

    Methods:
        unzip_and_extract_csv(source_directory, zip_path): Unzips the provided ZIP file and extracts the CSV file.
        load_data(file_name): Loads the data from the extracted CSV file.
        extract_data(data): Extracts neuron IDs, calcium traces, and time vector from the CSV data.
        create_metadata(): Creates a dictionary of extra information or metadata for the dataset.
        preprocess(): Preprocesses the Venkatachalam et al., 2024 neural data and saves a pickle file.
    """

    def __init__(self, transform, smooth_method, interpolate_method, resample_dt, **kwargs):
        """
        Initialize the Venkatachalam2024Preprocessor with the provided parameters.

        Parameters:
            transform (object): The sklearn transformation to be applied to the data.
            smooth_method (str): The smoothing method to apply to the data.
            interpolate_method (str): The interpolation method to use when resampling the data.
            resample_dt (float): The resampling time interval in seconds.
            **kwargs: Additional keyword arguments for smoothing.
        """
        super().__init__(
            "Venkatachalam2024",
            transform,
            smooth_method,
            interpolate_method,
            resample_dt,
            **kwargs,
        )
        self.citation = "Seyedolmohadesin, M, unpublished 2024, _Brain-wide neural activity data in C. elegans_. https://chemosensory-data.worm.world/ [Last Accessed: October 3, 2024]"

    def unzip_and_extract_csv(self, source_directory, zip_path):
        """
        Unzips the provided ZIP file and extracts the CSV file.

        Parameters:
            source_directory (str): The directory where the ZIP file is located.
            zip_path (str): The path to the ZIP file.

        Returns:
            str: The path to the extracted CSV file.
        """
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(source_directory)
        return zip_path.replace(".zip", ".csv")

    def load_data(self, file_name):
        """
        Loads the data from the extracted CSV file.

        Parameters:
            file_name (str): The name of the ZIP file containing the CSV data.

        Returns:
            pd.DataFrame: The loaded data as a pandas DataFrame.
        """
        zip_path = os.path.join(self.raw_data_path, self.source_dataset, file_name)
        csv_file = self.unzip_and_extract_csv(
            os.path.join(self.raw_data_path, self.source_dataset), zip_path
        )
        data = pd.read_csv(csv_file)
        return data

    def extract_data(self, data):
        """
        Extracts neuron IDs, calcium traces, and time vector from the CSV data.

        Parameters:
            data (pd.DataFrame): The loaded data as a pandas DataFrame.

        Returns:
            tuple: A tuple containing neuron IDs, calcium traces, and time vector.
        """
        # 9 columns + 98 columns of blank neural data
        time_vector = (
            data.columns[107:-1].astype(float).to_numpy() * 0.375
        )  # Columns 9 onwards contain calcium data with dt of 375ms
        traces = data.iloc[:, 107:-1].values.T  # transpose to get (time, neurons)
        # Remove neuron traces that are all NaN values
        mask = np.argwhere(~np.isnan(traces).all(axis=0)).flatten()
        traces = traces[:, mask]
        # Get the neuron labels corresponding to the traces
        neuron_ids = np.array(data["neuron"].unique(), dtype=str)[mask].tolist()
        # Impute any remaining NaN values
        # NOTE: This is very slow with the default settings!
        imputer = IterativeImputer(random_state=0, n_nearest_features=10, skip_complete=False)
        if np.isnan(traces).any():
            traces = imputer.fit_transform(traces)
        # Make the extracted data into a list of arrays
        all_IDs = [neuron_ids]
        all_traces = [traces]
        timeVectorSeconds = [time_vector]
        # Return the extracted data
        return all_IDs, all_traces, timeVectorSeconds

    def preprocess(self):
        """
        Preprocesses the Venkatachalam et al., 2024 neural data and saves it as a pickle file

        The data is read from ZIP files containing CSV data located in the dataset directory.

        Steps:
            1. Initialize an empty dictionary for preprocessed data and a worm index.
            2. Iterate through the ZIP files in the dataset directory:
                - Unzip and extract the CSV file.
                - Load the data from the CSV file.
                - Extract neuron IDs, calcium traces, and time vector from the CSV data.
                - Preprocess the traces and update the preprocessed data dictionary.
            3. Reshape the calcium data for each worm.
            4. Save the preprocessed data to a pickle file.
        """
        preprocessed_data = dict()
        worm_idx = 0
        for file_name in os.listdir(os.path.join(self.raw_data_path, self.source_dataset)):
            if not file_name.endswith(".zip"):
                continue
            raw_data = self.load_data(file_name)
            neuron_ids, traces, raw_time_vector = self.extract_data(raw_data)
            metadata = dict(
                citation=self.citation,
                data_file=os.path.join(
                    os.path.basename(self.raw_data_path), self.source_dataset, file_name
                ),
            )
            preprocessed_data, worm_idx = self.preprocess_traces(
                neuron_ids,
                traces,
                raw_time_vector,
                preprocessed_data,
                worm_idx,
                metadata,
            )
        for worm in preprocessed_data.keys():
            preprocessed_data[worm] = reshape_calcium_data(preprocessed_data[worm])
        self.save_data(preprocessed_data)
        # logger.info(f"Finished processing {self.source_dataset}.")
