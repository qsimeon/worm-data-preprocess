from preprocess._pkg import (
    NEURON_LABELS,
    RAW_DATA_DIR,
    ROOT_DIR,
    mat73,
    np,
    pd,
    pickle,
    os,
    torch,
    coalesce,
    to_dense_adj,
    Data,
    StandardScaler,
    List,
)
from preprocess.preprocessors._helpers import (
    smooth_data_preprocess,
    interpolate_data,
    aggregate_data,
    CausalNormalizer,
)


class ConnectomeBasePreprocessor:
    """
    Base class for preprocessing connectome data.

    This class provides common methods and attributes for preprocessing connectome data,
    including loading neuron labels, loading a neuron master sheet, and performing common
    preprocessing tasks such as creating graph tensors and saving them.

    Attributes:
        neuron_labels (List[str]): List of neuron labels.
        neuron_master_sheet (pd.DataFrame): DataFrame containing the neuron master sheet.
        neuron_to_idx (dict): Dictionary mapping neuron labels to their corresponding indices.

    Methods:
        load_neuron_labels() -> List[str]:
            Loads the neuron labels from a file or a constant.
        load_neuron_master_sheet() -> pd.DataFrame:
            Loads the neuron master sheet from a CSV file.
        preprocess_common_tasks(edge_index, edge_attr):
            Performs common preprocessing tasks such as creating graph tensors.
        save_graph_tensors(save_as: str, graph, num_classes, node_type, node_label, node_index, node_class):
            Saves the graph tensors to a file.
    """

    def __init__(self):
        """Initializes the ConnectomeBasePreprocessor with neuron labels and master sheet.

        This constructor initializes the ConnectomeBasePreprocessor by loading the neuron labels
        and the neuron master sheet. It also creates a dictionary mapping neuron labels to their
        corresponding indices.

        Attributes:
            neuron_labels (List[str]): List of neuron labels.
            neuron_master_sheet (pd.DataFrame): DataFrame containing the neuron master sheet.
            neuron_to_idx (dict): Dictionary mapping neuron labels to their corresponding indices.
        """
        self.neuron_labels = self.load_neuron_labels()
        self.neuron_master_sheet = self.load_neuron_master_sheet()
        self.neuron_to_idx = {
            label: idx for idx, label in enumerate(self.neuron_labels)
        }

    def load_neuron_labels(self) -> List[str]:
        """Loads the neuron labels from a file or a constant.

        Returns:
            List[str]: A list of neuron labels.
        """
        return NEURON_LABELS

    def load_neuron_master_sheet(self) -> pd.DataFrame:
        """Loads the neuron master sheet from a CSV file.

        Returns:
            pd.DataFrame: A DataFrame containing the neuron master sheet.
        """
        return pd.read_csv(os.path.join(RAW_DATA_DIR, "neuron_master_sheet.csv"))

    def preprocess_common_tasks(self, edge_index, edge_attr):
        """Performs common preprocessing tasks such as creating graph tensors.

        This function processes the edge indices and attributes to create graph tensors
        that represent the connectome. It ensures the correct mapping of neurons to their classes
        and types, checks for the symmetry of the gap junction adjacency matrix, and adds
        missing nodes with zero-weight edges to maintain graph completeness.

        Args:
            edge_index (torch.Tensor): Tensor containing the edge indices.
            edge_attr (torch.Tensor): Tensor containing the edge attributes.

        Returns:
            graph (torch_geometric.data.Data): The processed graph data object.
            node_type (torch.Tensor): Tensor of integers representing neuron types.
            node_label (dict): Dictionary mapping node indices to neuron labels.
            node_index (torch.Tensor): Tensor containing the node indices.
            node_class (dict): Dictionary mapping node indices to neuron classes.
            num_classes (int): The number of unique neuron classes.
        """
        # Filter the neuron master sheet to include only neurons present in the labels
        df_master = self.neuron_master_sheet[
            self.neuron_master_sheet["label"].isin(self.neuron_labels)
        ]

        # Create a position dictionary (pos) for neurons using their x, y, z coordinates
        pos_dict = df_master.set_index("label")[["x", "y", "z"]].to_dict("index")
        pos = {
            self.neuron_to_idx[label]: [
                pos_dict[label]["x"],
                pos_dict[label]["y"],
                pos_dict[label]["z"],
            ]
            for label in pos_dict
        }

        # Encode the neuron class (e.g., ADA, ADF) and create a mapping from node index to neuron class
        df_master["class"] = df_master["class"].fillna("Unknown")
        node_class = {
            self.neuron_to_idx[label]: neuron_class
            for label, neuron_class in zip(df_master["label"], df_master["class"])
        }
        num_classes = len(df_master["class"].unique())

        # Alphabetically sort neuron types and encode them as integers
        df_master["type"] = df_master["type"].fillna("Unknown")
        unique_types = sorted(
            df_master["type"].unique()
        )  # inter, motor, pharynx, sensory
        type_to_int = {neuron_type: i for i, neuron_type in enumerate(unique_types)}

        # Create tensor of neuron types (y) using the encoded integers
        y = torch.tensor(
            [type_to_int[neuron_type] for neuron_type in df_master["type"]],
            dtype=torch.long,
        )

        # Map node indices to neuron types using integers
        node_type = {
            self.neuron_to_idx[label]: type_to_int[neuron_type]
            for label, neuron_type in zip(df_master["label"], df_master["type"])
        }

        # Initialize the node features (x) as a tensor, here set as empty with 1024 features per node (customize as needed)
        x = torch.empty(len(self.neuron_labels), 1024, dtype=torch.float)

        # Create the mapping from node indices to neuron labels (e.g., 'ADAL', 'ADAR', etc.)
        node_label = {idx: label for label, idx in self.neuron_to_idx.items()}

        # Create the node index tensor for the graph
        node_index = torch.arange(len(self.neuron_labels))

        # Add missing nodes with zero-weight edges to ensure the adjacency matrix is 300x300
        all_indices = torch.arange(len(self.neuron_labels))
        full_edge_index = torch.combinations(all_indices, r=2).T
        existing_edges_set = set(map(tuple, edge_index.T.tolist()))

        additional_edges = []
        additional_edge_attr = []
        for edge in full_edge_index.T.tolist():
            if tuple(edge) not in existing_edges_set:
                additional_edges.append(edge)
                additional_edge_attr.append(
                    [0, 0]
                )  # Add a zero-weight edge for missing connections

        # If there are additional edges, add them to the edge_index and edge_attr tensors
        if additional_edges:
            additional_edges = torch.tensor(additional_edges).T
            additional_edge_attr = torch.tensor(additional_edge_attr, dtype=torch.float)
            edge_index = torch.cat([edge_index, additional_edges], dim=1)
            edge_attr = torch.cat([edge_attr, additional_edge_attr], dim=0)

        # Check for symmetry in the gap junction adjacency matrix (electrical synapses should be symmetric)
        gap_junctions = to_dense_adj(
            edge_index=edge_index, edge_attr=edge_attr[:, 0]
        ).squeeze(0)
        if not torch.allclose(gap_junctions.T, gap_junctions):
            raise AssertionError("The gap junction adjacency matrix is not symmetric.")

        # Create the graph data object with all the processed information
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        graph.pos = pos  # Add positional information to the graph object

        return graph, node_type, node_label, node_index, node_class, num_classes

    def save_graph_tensors(
        self,
        save_as: str,
        graph: Data,
        node_type: dict,
        node_label: dict,
        node_index: torch.Tensor,
        node_class: dict,
        num_classes: int,
    ):
        """
        Saves the graph tensors and additional attributes to a file.

        Args:
            save_as (str): Filename for the saved graph data.
            graph (Data): Processed graph data containing node features (`x`), edge indices (`edge_index`), edge attributes (`edge_attr`), node positions (`pos`), and optional node labels (`y`).
            node_type (dict): Maps node index to neuron type (e.g., sensory, motor).
            node_label (dict): Maps node index to neuron label (e.g., 'ADAL').
            node_index (torch.Tensor): Tensor of node indices.
            node_class (dict): Maps node index to neuron class (e.g., 'ADA').
            num_classes (int): Number of unique neuron types/classes.

        The graph tensors dictionary includes connectivity (`edge_index`), attributes (`edge_attr`), neuron positions (`pos`), features (`x`), and additional information such as node labels and types.
        """

        # Collect the graph data and additional attributes in a dictionary
        graph_tensors = {
            "edge_index": graph.edge_index,
            "edge_attr": graph.edge_attr,
            "pos": graph.pos,
            "x": graph.x,
            "y": graph.y,
            "node_type": node_type,
            "node_label": node_label,
            "node_class": node_class,
            "node_index": node_index,
            "num_classes": num_classes,
        }

        # Save the graph tensors to a file
        torch.save(
            graph_tensors,
            os.path.join(ROOT_DIR, "data", "processed", "connectome", save_as),
        )


class NeuralBasePreprocessor:
    """
    This is a base class used for preprocessing different types of neurophysiological datasets.

    The class provides a template for loading, extracting, smoothing, resampling, and
    normalizing neural data, as well as saving the processed data in a pickle format.
    Specific datasets can be processed by creating a new class that inherits from this base class
    and overriding the methods as necessary.

    Attributes:
        source_dataset (str): The specific source dataset to be preprocessed.
        transform (object): The sklearn transformation to be applied to the
        data.
        or after resampling
        smooth_method (str): The smoothing method to apply to the data.
        resample_dt (float): The resampling time interval in seconds.
        raw_data_path (str): The path where the raw dat is downloaded at.
        processed_data_apth (str): The path at which to save the processed dataset.

    Methods:
        load_data(): Method for loading the raw data.
        extract_data(): Method for extracting the neural data from the raw data.
        smooth_data(): Method for smoothing the neural data.
        resample_data(): Method for resampling the neural data.
        normalize_data(): Method for normalizing the neural data.
        save_data(): Method for saving the processed data to .pickle format.
        create_neuron_idx(): Method for extracting a neuron label to index mapping from the raw data.
        preprocess_traces(): Base method for preprocessing the calcium traces. Some datasets may require
                    additional preprocessing steps, in which case this method should be overridden.

    Note:
        This class is intended to be subclassed, not directly instantiated.
        Specific datasets should implement their own versions of the
        `load_data`,`extract_data`, `smooth_data`, `resample_data`, `normalize_data`,
        `create_metadata` `save_data`, and `preprocess` methods.

    Example:
        class SpecificDatasetPreprocessor(NeuralBasePreprocessor):
            def load_data(self):
                # Implement dataset-specific loading logic here.
    """

    def __init__(
        self,
        source_dataset,
        transform=CausalNormalizer(),  # StandardScaler() #PowerTransformer() #CausalNormalizer() #None
        smooth_method="none",
        interpolate_method="linear",
        resample_dt=0.1,
        **kwargs,
    ):
        """
        Initialize the NeuralBasePreprocessor with the provided parameters.

        Parameters:
            source_dataset (str): The name of the source dataset to be preprocessed.
            transform (object, optional): The sklearn transformation to be applied to the data. Default is CausalNormalizer().
            smooth_method (str, optional): The smoothing method to apply to the data. Default is "moving".
            interpolate_method (str, optional): The interpolation method to use when resampling the data. Default is "linear".
            resample_dt (float, optional): The resampling time interval in seconds. Default is 0.1.
            **kwargs: Additional keyword arguments for smoothing.
        """
        self.source_dataset = source_dataset
        self.transform = transform
        self.smooth_method = smooth_method
        self.interpolate_method = interpolate_method
        self.resample_dt = resample_dt
        self.smooth_kwargs = kwargs
        self.raw_data_path = os.path.join(ROOT_DIR, "data", "opensource_neural_data")
        self.processed_data_path = os.path.join(ROOT_DIR, "data/processed/neural")

    def smooth_data(self, data, time_in_seconds):
        """
        Smooth the data using the specified smoothing method.

        Parameters:
            data (np.ndarray): The input data to be smoothed.
            time_in_seconds (np.ndarray): The time vector corresponding to the input data.

        Returns:
            np.ndarray: The smoothed data.
        """
        return smooth_data_preprocess(
            data,
            time_in_seconds,
            self.smooth_method,
            **self.smooth_kwargs,
        )

    def resample_data(self, time_in_seconds, ca_data, upsample=True):
        """
        Resample the calcium data to the desired time steps.
        The input time vector and data matrix should be matched in time,
        and the resampled time vector and data matrix should also be matched.

        Parameters:
            time_in_seconds (np.ndarray): Time vector in seconds with shape (time, 1).
            ca_data (np.ndarray): Original, non-uniformly sampled calcium data with shape (time, neurons).
            upsample (bool, optional): Whether to sample at a higher frequency (i.e., with smaller dt). Default is True.

        Returns:
            np.ndarray, np.ndarray: Resampled time vector and calcium data.
        """
        assert time_in_seconds.shape[0] == ca_data.shape[0], (
            f"Input mismatch! Time vector length ({time_in_seconds.shape[0]}) "
            f"doesn't match data length ({ca_data.shape[0]})."
        )
        # Perform upsampling (interpolation) or downsampling (aggregation) as needed
        if upsample:
            interp_time, interp_ca = interpolate_data(
                time_in_seconds,
                ca_data,
                target_dt=self.resample_dt,
                method=self.interpolate_method,
            )
        else:
            # First upsample to a finer dt before downsampling
            interp_time, interp_ca = interpolate_data(
                time_in_seconds,
                ca_data,
                target_dt=self.resample_dt / 6,  # Finer granularity first
                method=self.interpolate_method,
            )
            # Then aggregate over intervals to match the desired dt
            interp_time, interp_ca = aggregate_data(
                interp_time,
                interp_ca,
                target_dt=self.resample_dt,
            )
        # Ensure the resampled time and data are the same shape
        if interp_time.shape[0] != interp_ca.shape[0]:
            raise ValueError(
                f"Resampling mismatch! Resampled time vector ({interp_time.shape[0]}) "
                f"doesn't match resampled data length ({interp_ca.shape[0]})."
            )
        return interp_time, interp_ca

    def normalize_data(self, data):
        """
        Normalize the data using the specified transformation.

        Parameters:
            data (np.ndarray): The input data to be normalized.

        Returns:
            np.ndarray: The normalized data.
        """
        if self.transform is None:
            return data
        return self.transform.fit_transform(data)

    def save_data(self, data_dict):
        """
        Save the processed data to a .pickle file.

        Parameters:
            data_dict (dict): The processed data to be saved.
        """
        file = os.path.join(self.processed_data_path, f"{self.source_dataset}.pickle")
        with open(file, "wb") as f:
            pickle.dump(data_dict, f)

    def create_neuron_idx(self, unique_IDs):
        """
        Create a neuron label to index mapping from the raw data.

        Parameters:
            unique_IDs (list): List of unique neuron IDs.

        Returns:
            dict: Mapping of neuron labels to indices.
            int: Number of labeled neurons.
        """
        # TODO: Supplement this this with the Leifer2023 version so that we only need this one definition.
        idx_to_neuron = {
            nid: (
                str(nid)
                if (
                    j is None
                    or isinstance(j, np.ndarray)
                    or j == "merge"
                    or not j.isalnum()
                )
                else str(j)
            )
            for nid, j in enumerate(unique_IDs)
        }
        idx_to_neuron = {
            nid: (
                name.replace("0", "")
                if not name.endswith("0") and not name.isnumeric()
                else name
            )
            for nid, name in idx_to_neuron.items()
        }
        idx_to_neuron = {
            nid: (str(nid) if name not in set(NEURON_LABELS) else name)
            for nid, name in idx_to_neuron.items()
        }
        neuron_to_idx = dict((v, k) for k, v in idx_to_neuron.items())
        num_labeled_neurons = len(
            [k for k in neuron_to_idx.keys() if not k.isnumeric()]
        )  # number of labled neurons
        return neuron_to_idx, num_labeled_neurons

    def find_nearest_label(self, query, possible_labels, char="?"):
        """
        Finds the nearest neuron label from a list given a query.

        Parameters:
            query (str): The query string containing the neuron label with ambiguity.
            possible_labels (list): The list of possible neuron labels.
            char (str, optional): The character representing ambiguity in the query. Default is "?".

        Returns:
            tuple: A tuple containing the nearest neuron label and its index in the possible labels list.
        """
        # Ensure the possible labels is a sorted list
        possible_labels = sorted(possible_labels)
        # Remove the '?' from the query to simplify comparison
        query_base = query.replace(char, "")
        # Initialize variables to track the best match
        nearest_label = None
        highest_similarity = -1  # Start with lowest similarity possible
        for label in possible_labels:
            # Count matching characters, ignoring the character at the position of '?'
            similarity = sum(1 for q, l in zip(query_base, label) if q == l)
            # Update the nearest label if this one is more similar
            if similarity > highest_similarity:
                nearest_label = label
                highest_similarity = similarity
        return nearest_label, possible_labels.index(nearest_label)

    def get_longest_nan_stretch(self, arr):
        """
        Calculate the longest continuous stretch of NaNs in a 1D array.

        Parameters:
        arr (np.array): 1D Input array to check.

        Returns:
        int: Length of the longest continuous stretch of NaNs.
        """
        assert arr.ndim == 1, "Array must be a 1D time series."
        isnan = np.isnan(arr)
        if not np.any(isnan):
            return 0
        stretches = np.diff(
            np.where(np.concatenate(([isnan[0]], isnan[:-1] != isnan[1:], [True])))[0]
        )[::2]
        return stretches.max() if len(stretches) > 0 else 0

    def filter_bad_traces_by_nan_stretch(self, data, nan_stretch_threshold=0.05):
        """
        Filters out traces with long stretches of NaNs.

        Parameters:
        data (np.array): The neural data array with shape (time_points, neurons).
        nan_stretch_threshold (float): Proportion of the total recording time above which traces are considered bad.

        Returns:
        (np.array, np.array): Tuple of filtered neural data and the associated mask into the original data array.
        """
        t, n = data.shape
        max_nan_stretch_allowed = int(t * nan_stretch_threshold)
        bad_traces_mask = (
            np.apply_along_axis(self.get_longest_nan_stretch, 0, data)
            > max_nan_stretch_allowed
        )
        good_traces_mask = ~bad_traces_mask
        filtered_data = data[:, good_traces_mask]
        return filtered_data, good_traces_mask

    def is_monotonic_linear(self, arr):
        """
        Checks if the array is a line with a constant slope (i.e., linear).

        Parameters:
            arr (np.ndarray): The input array to check.

        Returns:
            bool: True if the array is linear, False otherwise.
        """
        assert arr.ndim == 1, "Array must be a 1D (univariate) time series."
        diff = np.round(np.diff(arr), decimals=3)
        result = np.unique(diff)
        return result.size == 1

    def filter_bad_traces_by_linear_segments(
        self, data, window_size=50, linear_segment_threshold=1e-3
    ):
        """
        Filters out traces with significant proportions of linear segments. Linear segments suggest
        that the data was imputed with linear interpolation to remove stretches of NaN values.

        There are weird-looking traces in some raw data caused by interpolations of missing values
        (NaNs) when neurons were not consistently tracked over time due to imperfect nonrigid registration.
        This helper function was written to filter out these problematic imputed neural traces.

        Parameters:
        data (np.array): The neural data array with shape (time_points, neurons).
        window_size (int): The size of the window to check for linearity.
        linear_segment_threshold (float): Proportion of linear segments above which traces are considered bad.

        Returns:
        (np.array, nparray): Tuple of filtered neural data and the associated mask into the original data array.
        """
        t, n = data.shape
        linear_segments = np.zeros(n, dtype=int)
        window_start = range(
            0, t - window_size, window_size // 2
        )  # overlapping/staggered windows faster than non-overlapping
        for i in window_start:
            segment = data[i : i + window_size, :]
            ls = np.apply_along_axis(self.is_monotonic_linear, 0, segment)
            linear_segments += ls.astype(int)
        proportion_linear = linear_segments / len(window_start)
        bad_traces_mask = np.array(proportion_linear > linear_segment_threshold)
        good_traces_mask = ~bad_traces_mask
        filtered_data = data[:, good_traces_mask]
        return filtered_data, good_traces_mask

    def load_data(self, file_name):
        """
        Load the raw data from a .mat file.
        The  simple place-holder implementation seen here for the
        Skora, Kato, Nichols, Uzel, and Kaplan datasets but should
        be customized for the others.

        Parameters:
            file_name (str): The name of the file to load.

        Returns:
            dict: The loaded data.
        """
        return mat73.loadmat(
            os.path.join(self.raw_data_path, self.source_dataset, file_name),
            verbose=False,
        )

    def extract_data(self):
        """
        Extract the basic data (neuron IDs, calcium traces, and time vector) from the raw data file.
        This method should be overridden by subclasses to implement dataset-specific extraction logic.
        """
        raise NotImplementedError()

    def create_metadata(self, **kwargs):
        """
        Create a dictionary of extra information or metadata for a dataset.

        Returns:
            dict: A dictionary of extra information or metadata.
        """
        extra_info = dict()
        extra_info.update(kwargs)
        return extra_info

    def pick_non_none(self, l):
        """
        Returns the first non-None element in a list.

        Parameters:
            l (list): The input list.

        Returns:
            The first non-None element in the list.
        """
        for i in range(len(l)):
            if l[i] is not None:
                return l[i]
        return None

    def preprocess(self):
        """
        Main preprocessing method that calls the other methods in the class.
        This method should be overridden by subclasses to implement dataset-specific preprocessing logic.
        """
        raise NotImplementedError()
    
    def preprocess_traces(
        self,
        neuron_IDs,
        traces,
        raw_timeVectorSeconds,
        preprocessed_data,
        worm_idx,
        metadata=dict(),
    ):
        """
        Helper function for preprocessing calcium fluorescence neural data from
        one worm.

        Called immediately before saving the data for each worm in each subclass
        of NeuralBaseProcessor (dataset specific processors).

            1. Compute residual calcium on the raw data.
            2. Smooth the raw data and its residual.
            3. Resample the raw signals to the desired time grid.
            4. Normalize the resampled raw data.
            5. Name the worm and update its data.

        Parameters:
            neuron_IDs (list): List of arrays of neuron IDs.
            traces (list): List of arrays of calcium traces, with indices corresponding to neuron_IDs.
            raw_timeVectorSeconds (list): List of arrays of time vectors, with indices corresponding to neuron_IDs.
            preprocessed_data (dict): Dictionary of preprocessed data from previous worms.
            worm_idx (int): Index of the current worm.
            metadata (dict, optional): Additional metadata to include.

        Returns:
            tuple: (preprocessed_data, worm_idx) where preprocessed_data is the updated data dictionary
                and worm_idx is the next worm index.
        """
        assert len(neuron_IDs) == len(traces) == len(raw_timeVectorSeconds), (
            "Lists for neuron labels, activity data, and time vectors must all be the same length."
        )

        for i, trace_data in enumerate(traces):
            # Verify shape consistency: (time, neurons)
            assert trace_data.ndim == 2, "Calcium traces must be 2D arrays."
            assert trace_data.shape[0] == len(raw_timeVectorSeconds[i]), (
                "Calcium trace does not have the right number of time points."
            )
            assert trace_data.shape[1] == len(neuron_IDs[i]), (
                "Calcium trace does not have the right number of neurons."
            )

            # Skip empty or very short recordings
            if trace_data.size == 0 or len(raw_timeVectorSeconds[i]) < 600:
                continue

            # Map neuron IDs: pick non-None values and get unique neurons
            unique_IDs = [
                (self.pick_non_none(j) if isinstance(j, list) else j)
                for j in neuron_IDs[i]
            ]
            unique_IDs = [
                (str(_) if j is None or isinstance(j, np.ndarray) else str(j))
                for _, j in enumerate(unique_IDs)
            ]
            _, unique_indices = np.unique(unique_IDs, return_index=True)
            unique_IDs = [unique_IDs[idx] for idx in unique_indices]

            # Create mapping: neuron label -> index; skip if no labeled neurons
            neuron_to_idx, num_labeled_neurons = self.create_neuron_idx(unique_IDs)
            if num_labeled_neurons == 0:
                continue

            # Use only data for unique neurons
            trace_data = trace_data[:, unique_indices.astype(int)]

            # Prepare time vector and dt (original)
            time_in_seconds = raw_timeVectorSeconds[i].reshape(-1, 1).astype(np.float32)
            time_in_seconds -= time_in_seconds[0]  # start at 0.0 seconds
            dt = np.diff(time_in_seconds, axis=0, prepend=0.0)
            original_median_dt = np.median(dt[1:]).item()

            # Step 1: Compute residual calcium on raw data
            residual_calcium = np.gradient(
                trace_data, time_in_seconds.squeeze(), axis=0
            )
            original_residual_calcium = residual_calcium
            # Step 2: Smooth the raw data and its residual
            smooth_calcium_data = self.smooth_data(trace_data, time_in_seconds)
            original_smooth_calcium_data = smooth_calcium_data
            smooth_residual_calcium = self.smooth_data(
                residual_calcium, time_in_seconds
            )
            original_smooth_residual_calcium = smooth_residual_calcium
            # Decide on upsampling
            upsample = self.resample_dt < original_median_dt
            # Step 3: Resample raw signals
            _, resampled_raw_calcium_data = self.resample_data(
                time_in_seconds, trace_data, upsample
            )
            _, resampled_residual_calcium = self.resample_data(
                time_in_seconds, residual_calcium, upsample
            )
            resampled_time_in_seconds, resampled_smooth_calcium_data = (
                self.resample_data(time_in_seconds, smooth_calcium_data, upsample)
            )
            resampled_time_in_seconds -= resampled_time_in_seconds[0]
            _, resampled_smooth_residual_calcium = self.resample_data(
                time_in_seconds, smooth_residual_calcium, upsample
            )
            resampled_dt = np.diff(resampled_time_in_seconds, axis=0, prepend=0.0)
            resampled_median_dt = np.median(resampled_dt[1:]).item()
            # Step 4: Normalize the resampled (optionally smoothed) data
            norm_calcium_data = self.normalize_data(resampled_smooth_calcium_data)
            resampled_calcium_data = norm_calcium_data

            cumulative_data = {}
            if isinstance(self.transform, CausalNormalizer):
                cumulative_data = {
                    "cumulative_mean": self.transform.cumulative_mean_,
                    "cumulative_std": self.transform.cumulative_std_,
                }
                assert (
                    norm_calcium_data.shape == cumulative_data["cumulative_mean"].shape
                ), "Cumulative data is misshaped"

            # Validate that the resampled median dt matches the desired dt
            assert np.isclose(self.resample_dt, resampled_median_dt, atol=0.01), (
                f"Resampling failed. The median dt ({resampled_median_dt}) of the resampled time vector is different from desired dt ({self.resample_dt})."
            )

            # Determine final dimensions and number of unlabeled neurons
            max_timesteps, num_neurons = resampled_calcium_data.shape
            num_unlabeled_neurons = int(num_neurons) - num_labeled_neurons

            # ----- Step 5: Name the worm and update its data -----
            worm = "worm" + str(worm_idx)
            worm_idx += 1

            worm_dict = {
                worm: {
                    "calcium_data": resampled_calcium_data,  # resampled, smoothed, normalized
                    "source_dataset": self.source_dataset,
                    "dt": resampled_dt,  # vector from resampled time vector
                    "idx_to_neuron": {v: k for k, v in neuron_to_idx.items()},
                    "interpolate_method": self.interpolate_method,
                    "max_timesteps": int(
                        max_timesteps
                    ),  # scalar from resampled time vector
                    "median_dt": self.resample_dt,  # scalar from resampled time vector
                    "neuron_to_idx": neuron_to_idx,
                    "num_labeled_neurons": num_labeled_neurons,
                    "num_neurons": int(num_neurons),
                    "num_unlabeled_neurons": num_unlabeled_neurons,
                    "original_dt": dt,  # vector from original time vector
                    "original_calcium_data": trace_data,  # untouched
                    **cumulative_data,  # cumulative mean and std from CausalNormalizer
                    "original_max_timesteps": int(
                        trace_data.shape[0]
                    ),  # scalar from original time vector
                    "original_median_dt": original_median_dt,  # scalar from original time vector
                    "original_residual_calcium": original_residual_calcium,  # original (computed on normalized data in A, raw in B)
                    "original_smooth_calcium_data": original_smooth_calcium_data,  # normalized and smoothed (original)
                    "original_smooth_residual_calcium": original_smooth_residual_calcium,  # smoothed (original)
                    "original_time_in_seconds": time_in_seconds,  # original time vector
                    "residual_calcium": resampled_residual_calcium,  # resampled
                    "smooth_calcium_data": resampled_smooth_calcium_data,  # normalized, smoothed and resampled
                    "smooth_method": self.smooth_method,
                    "smooth_residual_calcium": resampled_smooth_residual_calcium,  # smoothed and resampled
                    "time_in_seconds": resampled_time_in_seconds,  # resampled time vector
                    "worm": worm,  # worm ID
                    "extra_info": self.create_metadata(
                        **metadata
                    ),  # additional metadata
                }
            }
            preprocessed_data.update(worm_dict)
        return preprocessed_data, worm_idx
