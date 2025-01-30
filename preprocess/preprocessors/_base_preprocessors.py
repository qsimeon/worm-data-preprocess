from preprocess._pkg import *
from preprocess.preprocessors._helpers import *

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
        self.neuron_to_idx = {label: idx for idx, label in enumerate(self.neuron_labels)}

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
        unique_types = sorted(df_master["type"].unique())  # inter, motor, pharynx, sensory
        type_to_int = {neuron_type: i for i, neuron_type in enumerate(unique_types)}

        # Create tensor of neuron types (y) using the encoded integers
        y = torch.tensor(
            [type_to_int[neuron_type] for neuron_type in df_master["type"]], dtype=torch.long
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
        gap_junctions = to_dense_adj(edge_index=edge_index, edge_attr=edge_attr[:, 0]).squeeze(0)
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
            graph_tensors, os.path.join(ROOT_DIR, "data", "processed", "connectome", save_as)
        )


class DefaultPreprocessor(ConnectomeBasePreprocessor):
    """
    Default preprocessor for connectome data.

    This class extends the ConnectomeBasePreprocessor to provide specific preprocessing
    steps for the default connectome data. It includes methods for loading, processing,
    and saving the connectome data.

    The default connectome data used here is a MATLAB preprocessed version of Cook et al., 2019 by
    Kamal Premaratne. If the raw data isn't found, please download the zip file from this link:
    https://wormwiring.org/matlab%20scripts/Premaratne%20MATLAB-ready%20files%20.zip,
    unzip the archive in the data/raw folder, then run the MATLAB script `export_nodes_edges.m`.

    Methods:
        preprocess(save_as="graph_tensors.pt"):
            Preprocesses the connectome data and saves the graph tensors to a file.
    """

    def preprocess(self, save_as="graph_tensors.pt"):
        """
        Preprocesses the connectome data and saves the graph tensors to a file.

        The data is read from multiple CSV files named "GHermChem_Edges.csv",
        "GHermChem_Nodes.csv", "GHermGap_Edges.csv", and "GHermGap_Nodes.csv".

        Args:
            save_as (str, optional): The name of the file to save the graph tensors to. Default is "graph_tensors.pt".

        Steps:
            1. Load the chemical synapse edges and nodes from "GHermChem_Edges.csv" and "GHermChem_Nodes.csv".
            2. Load the electrical synapse edges and nodes from "GHermGap_Edges.csv" and "GHermGap_Nodes.csv".
            3. Initialize sets for all C. elegans hermaphrodite neurons.
            4. Process the chemical synapse edges and nodes:
                - Filter edges and nodes based on neuron labels.
                - Append edges and attributes to the respective lists.
            5. Process the electrical synapse edges and nodes:
                - Filter edges and nodes based on neuron labels.
                - Append edges and attributes to the respective lists.
            6. Convert edge attributes and edge indices to tensors.
            7. Call the `preprocess_common_tasks` method to perform common preprocessing tasks.
            8. Save the graph tensors to the specified file.
        """
        # # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # # Override DefaultProprecessor with Witvliet2020Preprocessor7, a more up-to-date connectome of C. elegans.
        # return Witvliet2020Preprocessor7.preprocess(self, save_as="graph_tensors.pt")
        # # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # Names of all C. elegans hermaphrodite neurons
        neurons_all = set(self.neuron_labels)
        sep = r"[\t,]"

        # Chemical synapses nodes and edges
        GHermChem_Edges = pd.read_csv(
            os.path.join(RAW_DATA_DIR, "GHermChem_Edges.csv"), sep=sep
        )  # edges
        GHermChem_Nodes = pd.read_csv(
            os.path.join(RAW_DATA_DIR, "GHermChem_Nodes.csv"), sep=sep
        )  # nodes

        # Gap junctions
        GHermElec_Sym_Edges = pd.read_csv(
            os.path.join(RAW_DATA_DIR, "GHermElec_Sym_Edges.csv"), sep=sep
        )  # edges
        GHermElec_Sym_Nodes = pd.read_csv(
            os.path.join(RAW_DATA_DIR, "GHermElec_Sym_Nodes.csv"), sep=sep
        )  # nodes

        # Neurons involved in gap junctions
        df = GHermElec_Sym_Nodes
        df["Name"] = [
            v.replace("0", "") if not v.endswith("0") else v for v in df["Name"]
        ]  # standard naming
        Ggap_nodes = (
            df[df["Name"].isin(neurons_all)].sort_values(by=["Name"]).reset_index()
        )  # filter out non-neurons

        # Neurons (i.e. nodes) in chemical synapses
        df = GHermChem_Nodes
        df["Name"] = [
            v.replace("0", "") if not v.endswith("0") else v for v in df["Name"]
        ]  # standard naming
        Gsyn_nodes = (
            df[df["Name"].isin(neurons_all)].sort_values(by=["Name"]).reset_index()
        )  # filter out non-neurons

        # Gap junctions edges
        df = GHermElec_Sym_Edges
        df["EndNodes_1"] = [
            v.replace("0", "") if not v.endswith("0") else v for v in df["EndNodes_1"]
        ]
        df["EndNodes_2"] = [
            v.replace("0", "") if not v.endswith("0") else v for v in df["EndNodes_2"]
        ]
        inds = [
            i
            for i in GHermElec_Sym_Edges.index
            if df.iloc[i]["EndNodes_1"] in set(Ggap_nodes.Name)
            and df.iloc[i]["EndNodes_2"] in set(Ggap_nodes.Name)
        ]  # indices
        Ggap_edges = df.iloc[inds].reset_index(drop=True)

        # Chemical synapses
        df = GHermChem_Edges
        df["EndNodes_1"] = [
            v.replace("0", "") if not v.endswith("0") else v for v in df["EndNodes_1"]
        ]
        df["EndNodes_2"] = [
            v.replace("0", "") if not v.endswith("0") else v for v in df["EndNodes_2"]
        ]
        inds = [
            i
            for i in GHermChem_Edges.index
            if df.iloc[i]["EndNodes_1"] in set(Gsyn_nodes.Name)
            and df.iloc[i]["EndNodes_2"] in set(Gsyn_nodes.Name)
        ]  # indices
        Gsyn_edges = df.iloc[inds].reset_index(drop=True)

        # Map neuron names (IDs) to indices
        neuron_to_idx = dict(zip(self.neuron_labels, range(len(self.neuron_labels))))

        # edge_index for gap junctions
        arr = Ggap_edges[["EndNodes_1", "EndNodes_2"]].values
        ggap_edge_index = torch.empty(*arr.shape, dtype=torch.long)
        for i, row in enumerate(arr):
            ggap_edge_index[i, :] = torch.tensor([neuron_to_idx[x] for x in row], dtype=torch.long)
        ggap_edge_index = ggap_edge_index.T  # [2, num_edges]
        # Do the reverse direction to ensure symmetry of gap junctions
        ggap_edge_index = torch.hstack((ggap_edge_index, ggap_edge_index[[1, 0], :]))

        # edge_index for chemical synapses
        arr = Gsyn_edges[["EndNodes_1", "EndNodes_2"]].values
        gsyn_edge_index = torch.empty(*arr.shape, dtype=torch.long)
        for i, row in enumerate(arr):
            gsyn_edge_index[i, :] = torch.tensor([neuron_to_idx[x] for x in row], dtype=torch.long)
        gsyn_edge_index = gsyn_edge_index.T  # [2, num_edges]

        # edge attributes
        # NOTE: The first feature represents the weight of the gap junctions;
        # The second feature represents the weight of the chemical synapses.
        num_edge_features = 2

        # edge_attr for gap junctions
        num_edges = len(Ggap_edges)
        ggap_edge_attr = torch.empty(
            num_edges, num_edge_features, dtype=torch.float
        )  # [num_edges, num_edge_features]
        for i, weight in enumerate(Ggap_edges.Weight.values):
            ggap_edge_attr[i, :] = torch.tensor(
                [weight, 0], dtype=torch.float
            )  # electrical synapse encoded as [1,0]
        # Do the reverse direction to ensure symmetry of gap junctions
        ggap_edge_attr = torch.vstack((ggap_edge_attr, ggap_edge_attr))

        # edge_attr for chemical synapses
        num_edges = len(Gsyn_edges)
        gsyn_edge_attr = torch.empty(
            num_edges, num_edge_features, dtype=torch.float
        )  # [num_edges, num_edge_features]
        for i, weight in enumerate(Gsyn_edges.Weight.values):
            gsyn_edge_attr[i, :] = torch.tensor(
                [0, weight], dtype=torch.float
            )  # chemical synapse encoded as [0,1]

        # Merge electrical and chemical graphs into a single connectome graph
        combined_edge_index = torch.hstack((ggap_edge_index, gsyn_edge_index))
        combined_edge_attr = torch.vstack((ggap_edge_attr, gsyn_edge_attr))
        edge_index, edge_attr = coalesce(
            combined_edge_index, combined_edge_attr, reduce="add"
        )  # features = [elec_wt, chem_wt]

        # Perform common preprocessing tasks to create graph tensors
        graph, node_type, node_label, node_index, node_class, num_classes = (
            self.preprocess_common_tasks(edge_index, edge_attr)
        )

        # Save the processed graph tensors to the specified file
        self.save_graph_tensors(
            save_as,
            graph,
            node_type,
            node_label,
            node_index,
            node_class,
            num_classes,
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
        transform (object): The sklearn transformation to be applied to the data.
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
        # TODO: Try different transforms from sklearn such as QuantileTransformer, etc. as well as custom CausalNormalizer.
        transform=StandardScaler(),  # StandardScaler() #PowerTransformer() #CausalNormalizer() #None
        smooth_method="none",
        interpolate_method="linear",
        resample_dt=0.1,
        **kwargs,
    ):
        """
        Initialize the NeuralBasePreprocessor with the provided parameters.

        Parameters:
            source_dataset (str): The name of the source dataset to be preprocessed.
            transform (object, optional): The sklearn transformation to be applied to the data. Default is StandardScaler().
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
                if (j is None or isinstance(j, np.ndarray) or j == "merge" or not j.isalnum())
                else str(j)
            )
            for nid, j in enumerate(unique_IDs)
        }
        idx_to_neuron = {
            nid: (
                name.replace("0", "") if not name.endswith("0") and not name.isnumeric() else name
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
            np.apply_along_axis(self.get_longest_nan_stretch, 0, data) > max_nan_stretch_allowed
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
        return mat73.loadmat(os.path.join(self.raw_data_path, self.source_dataset, file_name), verbose=False)

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
        Helper function for preprocessing calcium fluorescence neural data from one worm.
        This method checks that the neuron labels, data matrix and time vector are of consistent
        shapes (e.g. number of timesteps in data matrix should be same as length of time vector).
        Any empty data (e.g. no labeled neurons or no recorded activity data) are thrown out.

        Parameters:
            neuron_IDs (list): List of arrays of neuron IDs.
            traces (list): List of arrays of calcium traces, with indices corresponding to neuron_IDs.
            raw_timeVectorSeconds (list): List of arrays of time vectors, with indices corresponding to neuron_IDs.
            preprocessed_data (dict): Dictionary of preprocessed data from previous worms that gets extended with more worms here.
            worm_idx (int): Index of the current worm.

        Returns:
            dict: Collection of all preprocessed worm data so far.
            int: Index of the next worm to preprocess.

        Steps:
            Iterate through the traces and preprocess each one:
                1. Normalize the calcium data.
                2. Compute the residual calcium.
                3. Smooth the data.
                4. Resample the data.
                5. Name the worm and update the index.
            Save the resulting data.
        """
        assert (
            len(neuron_IDs) == len(traces) == len(raw_timeVectorSeconds)
        ), "Lists for neuron labels, activity data, and time vectors must all be the same length."
        # Each worm has a unique set of neurons, time vectors and calcium traces
        for i, trace_data in enumerate(traces):
            # Matrix `trace_data` should be shaped as (time, neurons)
            assert trace_data.ndim == 2, "Calcium traces must be 2D arrays."
            assert trace_data.shape[0] == len(
                raw_timeVectorSeconds[i]
            ), "Calcium trace does not have the right number of time points."
            assert trace_data.shape[1] == len(
                neuron_IDs[i]
            ), "Calcium trace does not have the right number of neurons."
            # Ignore any worms with empty traces
            if trace_data.size == 0:
                continue
            # Ignore any worms with very short recordings
            if len(raw_timeVectorSeconds[i]) < 600:
                continue
            # Map labeled neurons
            unique_IDs = [
                (self.pick_non_none(j) if isinstance(j, list) else j) for j in neuron_IDs[i]
            ]
            unique_IDs = [
                (str(_) if j is None or isinstance(j, np.ndarray) else str(j))
                for _, j in enumerate(unique_IDs)
            ]
            _, unique_indices = np.unique(unique_IDs, return_index=True)
            unique_IDs = [unique_IDs[_] for _ in unique_indices]
            # Create neuron label to index mapping
            neuron_to_idx, num_labeled_neurons = self.create_neuron_idx(unique_IDs)
            # Ignore any worms with no labelled neurons
            if num_labeled_neurons == 0:
                continue
            # Only get data for unique neurons
            trace_data = trace_data[:, unique_indices.astype(int)]
            # Normalize calcium data
            calcium_data = self.normalize_data(trace_data)  # matrix
            # Compute residual calcium
            time_in_seconds = raw_timeVectorSeconds[i].reshape(raw_timeVectorSeconds[i].shape[0], 1)
            time_in_seconds = np.array(time_in_seconds, dtype=np.float32)  # vector
            time_in_seconds = time_in_seconds - time_in_seconds[0]  # start at 0.0 seconds
            dt = np.diff(time_in_seconds, axis=0, prepend=0.0)  # vector
            original_median_dt = np.median(dt[1:]).item()  # scalar
            residual_calcium = np.gradient(
                calcium_data, time_in_seconds.squeeze(), axis=0
            )  # vector
            # Smooth data
            smooth_calcium_data = self.smooth_data(calcium_data, time_in_seconds)
            smooth_residual_calcium = self.smooth_data(residual_calcium, time_in_seconds)
            # Resample data
            upsample = self.resample_dt < original_median_dt  # bool
            _, resampled_calcium_data = self.resample_data(time_in_seconds, calcium_data, upsample)
            _, resampled_residual_calcium = self.resample_data(
                time_in_seconds, residual_calcium, upsample
            )
            # NOTE: We use the resampling of the smooth calcium data to give us the resampled time points
            resampled_time_in_seconds, resampled_smooth_calcium_data = self.resample_data(
                time_in_seconds, smooth_calcium_data, upsample
            )
            resampled_time_in_seconds = (
                resampled_time_in_seconds - resampled_time_in_seconds[0]
            )  # start at 0.0 seconds
            _, resampled_smooth_residual_calcium = self.resample_data(
                time_in_seconds, smooth_residual_calcium, upsample
            )
            resampled_dt = np.diff(resampled_time_in_seconds, axis=0, prepend=0.0)  # vector
            resampled_median_dt = np.median(resampled_dt[1:]).item()  # scalar
            assert np.isclose(
                self.resample_dt, resampled_median_dt, atol=0.01
            ), f"Resampling failed. The median dt ({resampled_median_dt}) of the resampled time vector is different from desired dt ({self.resample_dt})."
            max_timesteps, num_neurons = resampled_calcium_data.shape
            num_unlabeled_neurons = int(num_neurons) - num_labeled_neurons
            # Name worm and update index
            worm = "worm" + str(worm_idx)  # use global worm index
            worm_idx += 1  # increment worm index
            # Save data
            worm_dict = {
                worm: {
                    "calcium_data": resampled_calcium_data,  # normalized and resampled
                    "source_dataset": self.source_dataset,
                    "dt": resampled_dt,  # vector from resampled time vector
                    "idx_to_neuron": dict((v, k) for k, v in neuron_to_idx.items()),
                    "interpolate_method": self.interpolate_method,
                    "max_timesteps": int(max_timesteps),  # scalar from resampled time vector
                    "median_dt": self.resample_dt,  # scalar from resampled time vector
                    "neuron_to_idx": neuron_to_idx,
                    "num_labeled_neurons": num_labeled_neurons,
                    "num_neurons": int(num_neurons),
                    "num_unlabeled_neurons": num_unlabeled_neurons,
                    "original_dt": dt,  # vector from original time vector
                    "original_calcium_data": calcium_data,  # normalized
                    "original_max_timesteps": int(
                        calcium_data.shape[0]
                    ),  # scalar from original time vector
                    "original_median_dt": original_median_dt,  # scalar from original time vector
                    "original_residual_calcium": residual_calcium,  # original
                    "original_smooth_calcium_data": smooth_calcium_data,  # normalized and smoothed
                    "original_smooth_residual_calcium": smooth_residual_calcium,  # smoothed
                    "original_time_in_seconds": time_in_seconds,  # original time vector
                    "residual_calcium": resampled_residual_calcium,  # resampled
                    "smooth_calcium_data": resampled_smooth_calcium_data,  # normalized, smoothed and resampled
                    "smooth_method": self.smooth_method,
                    "smooth_residual_calcium": resampled_smooth_residual_calcium,  # smoothed and resampled
                    "time_in_seconds": resampled_time_in_seconds,  # resampled time vector
                    "worm": worm,  # worm ID
                    "extra_info": self.create_metadata(
                        **metadata
                    ),  # additional information and metadata
                }
            }
            # Update preprocessed data collection
            preprocessed_data.update(worm_dict)
        # Return the updated preprocessed data and worm index
        return preprocessed_data, worm_idx


