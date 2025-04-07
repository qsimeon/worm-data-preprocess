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
    to_dense_adj,
    coalesce,
    Data,
    StandardScaler,
    List,
    Dict,
    Tuple,
    Set,
    defaultdict
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

    def _symmetrize_gap_junctions(
        self,
        gap_data: List[Tuple[int, int, float]],
        aggregation_method: str = "mean",
    ) -> Tuple[List[List[int]], List[List[float]]]:
        """
        Helper function 1 for dataset-specific preprocessors:
        
        Processes raw gap junction data to enforce symmetry.

        Aggregates weights for each undirected pair (or self-loop), calculates
        a representative symmetric weight, and creates explicit directed edges
        (i->j and j->i) with this weight. Warns if asymmetry is detected
        during aggregation based on the raw input weights for a pair.

        Args:
            gap_data: A list of tuples, where each tuple represents a raw gap
                      junction reading: (pre_neuron_idx, post_neuron_idx, weight).
            aggregation_method: How to combine multiple weights found for the
                                same undirected pair ('mean', 'sum', 'max'). Defaults to 'mean'.

        Returns:
            A tuple containing:
            - edges_gap_sym_list (List[List[int]]): List of symmetric edge pairs [[idx1, idx2], [idx2, idx1], ...].
            - edge_attr_gap_sym_list (List[List[float]]): Corresponding attributes [[weight, 0.0], [weight, 0.0], ...].
        """
        processed_gap_pairs: Dict[frozenset, List[float]] = defaultdict(list)
        edges_gap_sym_list = []
        edge_attr_gap_sym_list = []

        # 1. Aggregate weights per undirected pair/self-loop
        for idx1, idx2, weight in gap_data:
            # Basic validation: Ensure indices are within bounds
            # Note: neuron_to_idx mapping should happen before calling this helper
            if not (
                0 <= idx1 < len(self.neuron_labels)
                and 0 <= idx2 < len(self.neuron_labels)
            ):
                print(
                    f"Warning ({self.__class__.__name__}): Skipping gap edge with invalid indices ({idx1}, {idx2}). Max index is {len(self.neuron_labels) - 1}."
                )
                continue
            if weight == 0:
                continue  # Skip zero-weight entries -- added in common_tasks

            pair_key = frozenset({idx1, idx2})
            processed_gap_pairs[pair_key].append(float(weight))

        # 2. Create symmetric edges using aggregated weights
        for pair_key, weights_list in processed_gap_pairs.items():
            if not weights_list:
                continue  # Should not happen with defaultdict(list)

            # Calculate representative weight
            if aggregation_method == "sum":
                rep_weight = np.sum(weights_list)
            elif aggregation_method == "max":
                rep_weight = np.max(weights_list)
            else:  # Defaults to mean -- RECOMMENDED
                rep_weight = np.mean(weights_list)

            # Check for asymmetry in original data points for this pair/loop
            # (Compares weights before aggregation method was applied)
            if len(weights_list) > 1 and not np.allclose(
                weights_list[0], weights_list, atol=1e-6
            ):
                indices = list(pair_key)
                node_names = [
                    self.idx_to_label.get(idx, f"Idx {idx}") for idx in indices
                ]
                print(
                    f"  WARNING ({self.__class__.__name__}): Potential asymmetry detected in raw gap weights for pair/loop involving {node_names} (Indices: {indices}). "
                    f"Raw weights found: {weights_list}. Using {aggregation_method}: {rep_weight:.4f}"
                )

            # Create edges
            if len(pair_key) == 1:  # Self-loop
                idx = list(pair_key)[0]
                edges_gap_sym_list.append([idx, idx])
                edge_attr_gap_sym_list.append([rep_weight, 0.0])  # [gap, chem]
            elif len(pair_key) == 2:  # Regular pair
                idx1, idx2 = list(pair_key)
                edges_gap_sym_list.append([idx1, idx2])
                edge_attr_gap_sym_list.append([rep_weight, 0.0])
                edges_gap_sym_list.append([idx2, idx1])
                edge_attr_gap_sym_list.append([rep_weight, 0.0])

        return edges_gap_sym_list, edge_attr_gap_sym_list

    def _process_and_coalesce_edges(
        self,
        chem_data: List[Tuple[int, int, float]],
        gap_data: List[Tuple[int, int, float]],
        gap_aggregation_method: str = "mean",
        device: torch.device = torch.device("cpu"), 
        dtype: torch.dtype = torch.float,  
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Processes raw chemical and gap junction data, enforces gap symmetry,
        combines them, and returns coalesced edge index and attributes.

        Args:
            chem_data: List of chemical synapse tuples: (pre_idx, post_idx, weight).
            gap_data: List of raw gap junction tuples: (pre_idx, post_idx, weight).
            gap_aggregation_method: Method for gap weight aggregation ('mean', 'sum', 'max').
            device: The torch device for the output tensors.
            dtype: The torch dtype for the edge attributes.

        Returns:
            A tuple containing:
            - edge_index (torch.Tensor): Coalesced edge indices.
            - edge_attr (torch.Tensor): Coalesced edge attributes ([gap, chem]).
        """
        num_nodes = len(self.neuron_labels)
        if num_nodes == 0:
            # Return empty tensors if no nodes defined (should be caught earlier ideally)
            return torch.empty((2, 0), dtype=torch.long, device=device), torch.empty(
                (0, 2), dtype=dtype, device=device
            )

        # 1. Process Gap Junctions (Symmetrize)
        edges_gap_sym_list, edge_attr_gap_sym_list = self._symmetrize_gap_junctions(
            gap_data, aggregation_method=gap_aggregation_method
        )

        # 2. Process Chemical Synapses
        edges_chem_list = []
        edge_attr_chem_list = []
        processed_chem_pairs: Set[Tuple[int, int]] = set()
        for idx1, idx2, weight in chem_data:
            # Basic validation
            if not (0 <= idx1 < num_nodes and 0 <= idx2 < num_nodes):
                print(
                    f"Warning ({self.__class__.__name__}): Skipping chemical edge with invalid indices ({idx1}, {idx2})."
                )
                continue
            if weight == 0:
                continue  # Skip zero weights

            # Check for simple duplicates in input chemical list (good practice)
            current_pair = (idx1, idx2)
            if current_pair in processed_chem_pairs:
                # This indicates the child could have coalesced chemicals beforehand too
                # For now, we just take the first one encountered, but could aggregate here if needed
                print(f"  Note ({self.__class__.__name__}): Duplicate chemical edge {current_pair} found in input list. Using first occurrence.")
                continue
            processed_chem_pairs.add(current_pair)

            edges_chem_list.append([idx1, idx2])
            edge_attr_chem_list.append([0.0, float(weight)])  # [gap, chem]

        # 3. Convert to Tensors (Handle empty lists)
        if edges_gap_sym_list:
            ggap_edge_index_sym = (
                torch.tensor(edges_gap_sym_list, dtype=torch.long, device=device)
                .t()
                .contiguous()
            )
            ggap_edge_attr_sym = torch.tensor(
                edge_attr_gap_sym_list, dtype=dtype, device=device
            )
        else:
            ggap_edge_index_sym = torch.empty((2, 0), dtype=torch.long, device=device)
            ggap_edge_attr_sym = torch.empty((0, 2), dtype=dtype, device=device)

        if edges_chem_list:
            gsyn_edge_index = (
                torch.tensor(edges_chem_list, dtype=torch.long, device=device)
                .t()
                .contiguous()
            )
            gsyn_edge_attr = torch.tensor(
                edge_attr_chem_list, dtype=dtype, device=device
            )
        else:
            gsyn_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            gsyn_edge_attr = torch.empty((0, 2), dtype=dtype, device=device)

        # 4. Combine and Coalesce
        combined_edge_index = torch.hstack((ggap_edge_index_sym, gsyn_edge_index))
        combined_edge_attr = torch.vstack((ggap_edge_attr_sym, gsyn_edge_attr))

        if combined_edge_index.shape[1] > 0:
            # coalesce sums attributes for edges defined in both gap and chemical lists
            edge_index, edge_attr = coalesce(
                combined_edge_index,
                combined_edge_attr,
                num_nodes=num_nodes,
                reduce="add",  # Sums [gap, 0] + [0, chem] -> [gap, chem]
            )
        else:
            # Handle case where no edges exist at all
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            edge_attr = torch.empty(
                (0, 2), dtype=dtype, device=device
            )  # Ensure 2 features

        return edge_index, edge_attr

    def _preprocess_common_tasks(self, edge_index_in, edge_attr_in):
        """Performs common preprocessing tasks including input validation and graph completion.

        This function first validates the input tensors from the child preprocessor,
        specifically checking for duplicate edges with conflicting attributes. It enforces
        the contract that child preprocessors should provide coalesced edge data
        (i.e. not [gap weight, 0] and [0, chemical weight] as separate edges).
        It then processes the edge indices and attributes to create graph tensors
        that represent the connectome, ensuring graph completeness by adding missing
        edges with zero-weight attributes, mapping neurons to classes/types,
        and checking gap junction symmetry on the final completed graph.

        Args:
            edge_index_in (torch.Tensor) - (2 x # edges): Tensor containing edge
            indices from the specific preprocessor ([pre_neuron_idx, post_neuron_idx]).
            Expected to be coalesced (no duplicate pairs).
            edge_attr_in (torch.Tensor) - (2 x # edges): Tensor containing edge
            attributes ([gap, chem]) from the specific preprocessor.

        Returns:
            graph (torch_geometric.data.Data): The processed graph data object.
            node_type (dict): Dictionary mapping node indices to neuron type integers.
            node_label (dict): Dictionary mapping node indices to neuron labels.
            node_index (torch.Tensor): Tensor containing the node indices.
            node_class (dict): Dictionary mapping node indices to neuron classes.
            num_classes (int): The number of unique neuron classes.

        Raises:
            ValueError: If input tensors have mismatched shapes.
            ValueError: If `edge_index_in` contains duplicate edge pairs with
                        different attributes, violating the expectation of
                        coalesced input.
            AssertionError: If the gap junction adjacency matrix of the completed graph
                            is not symmetric.
        """
        num_nodes = len(self.neuron_labels)
        if num_nodes == 0:
            raise ValueError(
                f"Cannot process with zero neurons defined in self.neuron_labels in {self.__class__.__name__}."
            )
        if edge_attr_in.shape[0] != edge_index_in.shape[1]:
            raise ValueError(
                f"Input edge_index and edge_attr dimensions do not match in {self.__class__.__name__}."
            )
        if edge_attr_in.shape[0] > 0 and edge_attr_in.shape[1] != 2:
            print(
                f"Warning: Input edge_attr shape is {edge_attr_in.shape}, expected [num_edges, 2]. Assuming order is [gap, chem]."
            )
            
        # --- Validate Input: Check for duplicate edges with conflicting attributes ---
        processed_edges = {}  # Store the first attribute encountered for each edge
        # reverse map for clearer error messages
        idx_to_label = {idx: label for label, idx in self.neuron_to_idx.items()}

        # move to cpu to speed up
        edge_index_in_cpu = edge_index_in.cpu()
        edge_attr_in_cpu = edge_attr_in.cpu()

        for i in range(edge_index_in_cpu.shape[1]):
            u, v = edge_index_in_cpu[0, i].item(), edge_index_in_cpu[1, i].item()
            current_pair = (u, v)
            current_attr = edge_attr_in_cpu[i]

            if current_pair in processed_edges:
                previous_attr = processed_edges[current_pair]
                # Check if the attributes are actually different
                if not torch.equal(current_attr, previous_attr):
                    u_label = idx_to_label.get(u, f"Idx {u}")
                    v_label = idx_to_label.get(v, f"Idx {v}")
                    error_msg = (
                        f"Input Error in {self.__class__.__name__}: Duplicate edge "
                        f"({u_label} -> {v_label}) found with conflicting attributes. "
                        f"This indicates the input was not properly coalesced.\n"
                        f"  First occurrence attribute: {previous_attr.tolist()}\n"
                        f"  Conflicting attribute found: {current_attr.tolist()}\n"
                        f"Please ensure the child preprocessor '{self.__class__.__name__}' calls "
                        f"'coalesce(..., reduce=\"add\")' before passing edges to "
                        f"'preprocess_common_tasks'."
                    )
                    raise ValueError(error_msg)
                else: # Optional: Warn about redundant identical edges
                    print(f"Warning ({self.__class__.__name__}): Redundant identical edge "
                          f"({u_label} -> {v_label}) with attribute {current_attr.tolist()} found in input.")
                    pass
            else:
                processed_edges[current_pair] = current_attr

        # --- Ensure graph completeness and check symmetry ---
        # Now 'processed_edges' contains unique edges from the input and their first seen attribute.
        # We use this dictionary for the completion step.

        final_edge_list = []
        final_attr_list = []
        # Determine default dtype and device from input or default to float/cpu
        attr_dtype = edge_attr_in.dtype if edge_attr_in.shape[0] > 0 else torch.float
        attr_device = (
            edge_attr_in.device if edge_attr_in.shape[0] > 0 else torch.device("cpu")
        )
        index_device = (
            edge_index_in.device if edge_index_in.shape[0] > 0 else torch.device("cpu")
        )
        # Ensure zero_attr has 2 features if possible, matching expected edge_attr format
        zero_attr_features = (
            2
            if edge_attr_in.shape[0] == 0 or edge_attr_in.shape[1] == 2
            else edge_attr_in.shape[1]
        )
        # zero_attr creates the default [0,0] value for unspecified edges
        zero_attr = torch.zeros(
            zero_attr_features, dtype=attr_dtype, device=attr_device
        )

        for i in range(num_nodes):
            for j in range(num_nodes):
                current_pair = (i, j)
                final_edge_list.append([i, j])
                # Use the validated 'processed_edges' dictionary
                attr = processed_edges.get(current_pair, zero_attr) # default to [0,0]
                final_attr_list.append(attr)

        # Convert final lists to tensors
        if final_edge_list:
            edge_index = (
                torch.tensor(final_edge_list, dtype=torch.long, device=index_device)
                .t()
                .contiguous()
            )
            # Ensure stacking happens correctly even if lists contain tensors already
            if isinstance(final_attr_list[0], torch.Tensor):
                edge_attr = torch.stack(final_attr_list, dim=0)
            else:  # Should not happen if zero_attr is tensor and processed_edges stores tensors
                edge_attr = torch.tensor(
                    final_attr_list, dtype=attr_dtype, device=attr_device
                )

        else:  # Should not happen if num_nodes > 0
            edge_index = torch.empty((2, 0), dtype=torch.long, device=index_device)
            attr_features = 2  # Default to 2 features for empty case consistency
            edge_attr = torch.empty(
                (0, attr_features), dtype=attr_dtype, device=attr_device
            )

        # Check for symmetry in the gap junction adjacency matrix (electrical synapses should be symmetric)
        # This check is performed on the completed graph representation
        if (
            edge_attr.shape[0] > 0 and edge_attr.shape[1] > 0
        ):  # Check if attributes exist and have features
            try:
                gap_junctions = to_dense_adj(
                    edge_index=edge_index,
                    edge_attr=edge_attr[:, 0],  # Use gap weights (index 0)
                    max_num_nodes=num_nodes,
                ).squeeze(0)

                if not torch.allclose(gap_junctions, gap_junctions.T, atol=1e-6):
                    # Add specific warning about asymmetry location if needed
                    diff = torch.abs(gap_junctions - gap_junctions.T)
                    asym_indices = torch.nonzero(diff > 1e-6)
                    if asym_indices.shape[0] > 0:
                        r, c = asym_indices[0].tolist()
                        val1 = gap_junctions[r, c].item()
                        val2 = gap_junctions[c, r].item()
                        node_label_map = {
                            idx: label for label, idx in self.neuron_to_idx.items()
                        }  # Temp map needed here
                        node1 = node_label_map.get(r, f"Idx {r}")
                        node2 = node_label_map.get(c, f"Idx {c}")
                        print(
                            f"Symmetry Check WARNING: Asymmetry found between {node1} and {node2}. Gap({node1}->{node2})={val1:.4f}, Gap({node2}->{node1})={val2:.4f}"
                        )
                    raise AssertionError(f"The gap junction adjacency matrix (after completion) is not symmetric in {self.__class__.__name__}")
            except IndexError:
                print(
                    "Warning: Could not perform symmetry check - edge_attr might not have expected shape."
                )
        elif edge_attr.shape[0] > 0 and edge_attr.shape[1] == 0:
            print("Warning: Skipping symmetry check - edge_attr has 0 features.")


        # --- Process metadata using the completed graph ---
        # Filter the neuron master sheet to include only neurons present in the labels
        df_master = self.neuron_master_sheet[
            self.neuron_master_sheet["label"].isin(self.neuron_labels)
        ].copy()  # Use .copy() to avoid SettingWithCopyWarning if modifying df_master

        # Create a position dictionary (pos) for neurons using their x, y, z coordinates
        # Use self.neuron_to_idx for mapping labels to the graph indices
        pos_dict = df_master.set_index("label")[["x", "y", "z"]].to_dict("index")
        # Create pos tensor matching graph node order and device
        pos_tensor_data = torch.zeros(
            (num_nodes, 3), dtype=torch.float, device=edge_index.device
        )
        for label, idx in self.neuron_to_idx.items():
            if label in pos_dict:
                pos_tensor_data[idx] = torch.tensor(
                    [pos_dict[label]["x"], pos_dict[label]["y"], pos_dict[label]["z"]],
                    dtype=torch.float,
                )
            # else: keep as zeros or handle missing position data

        # Encode the neuron class (e.g., ADA, ADF) and create a mapping from node index to neuron class
        df_master["class"] = df_master["class"].fillna("Unknown")
        # Check for label existence
        node_class = {
            self.neuron_to_idx[label]: neuron_class
            for label, neuron_class in zip(df_master["label"], df_master["class"])
            if label in self.neuron_to_idx
        }
        num_classes = len(df_master["class"].unique())

        # Alphabetically sort neuron types and encode them as integers
        df_master["type"] = df_master["type"].fillna("Unknown")
        unique_types = sorted(df_master["type"].unique())
        type_to_int = {neuron_type: i for i, neuron_type in enumerate(unique_types)}

        # Create tensor of neuron types (y) using the encoded integers, matching graph node order
        y_list = [
            type_to_int.get("Unknown", 0)
        ] * num_nodes  # Default to 'Unknown' or 0
        type_map = dict(zip(df_master["label"], df_master["type"]))
        for label, idx in self.neuron_to_idx.items():
            neuron_type = type_map.get(label, "Unknown")
            y_list[idx] = type_to_int.get(neuron_type, type_to_int.get("Unknown", 0))
        y = torch.tensor(y_list, dtype=torch.long, device=edge_index.device)

        # Map node indices to neuron types using integers (for the returned dict)
        node_type = {idx: y_list[idx] for idx in range(num_nodes)}

        # Initialize the node features (x) as a tensor, matching device
        x = torch.empty(
            num_nodes, 1024, dtype=torch.float, device=edge_index.device
        ) 

        # Create the mapping from node indices to neuron labels (e.g., 'ADAL', 'ADAR', etc.)
        node_label = {idx: label for label, idx in self.neuron_to_idx.items()}

        # Create the node index tensor for the graph
        node_index = torch.arange(num_nodes, device=edge_index.device)

        # Create the graph data object with all the processed information
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        graph.pos = pos_tensor_data  # Add positional information as tensor

        return graph, node_type, node_label, node_index, node_class, num_classes

    def _save_graph_tensors(
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
            normalization_method = "standard"
            if isinstance(self.transform, CausalNormalizer):
                cumulative_data = {
                    "cumulative_mean": self.transform.cumulative_mean_,
                    "cumulative_std": self.transform.cumulative_std_,
                }
                normalization_method = "causal"
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
                    "normalization_method": normalization_method,
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
