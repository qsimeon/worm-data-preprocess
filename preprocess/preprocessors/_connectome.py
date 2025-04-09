"""
Contains dataset-specific connectome preprocessors,
which are subclasses of ConnectomeBasePreprocessor.

Count: 10
"""
from preprocess._pkg import *
from preprocess.preprocessors._helpers import *
from preprocess.preprocessors._base_preprocessors import ConnectomeBasePreprocessor



class DefaultPreprocessor(ConnectomeBasePreprocessor):
    """
    Default preprocessor for connectome data (non-dataset specific)

    This class extends the ConnectomeBasePreprocessor to provide specific preprocessing
    steps for the default connectome data. It includes methods for loading, processing,
    and saving the connectome data.

    The default connectome data used here is a MATLAB preprocessed version of Cook et al., 2019 by
    Kamal Premaratne. If the raw data isn't found, please download the zip file from this link:
    https://wormwiring.org/matlab%20scripts/Premaratne%20MATLAB-ready%20files%20.zip,
    unzip the archive in the data/raw folder, then run the MATLAB script `export_nodes_edges.m`.

    Methods:
        preprocess(save_as="graph_tensors_default.pt"):
            Preprocesses the connectome data and saves the graph tensors to a file.
    """

    def preprocess(self, save_as="graph_tensors_default.pt"):
        """
        Loads default connectome data from CSVs, processes using base class helpers, and saves tensors.

        Reads chemical and gap junction data from GHerm* CSV files, standardizes
        neuron names, filters based on known neuron lists, collects raw edge tuples,
        calls base helpers to symmetrize gaps and coalesce edges, then processes
        common tasks and saves the final graph.

        Args:
            save_as (str, optional): Filename for saved tensors. Default "graph_tensors_default.pt".
        """
        # Override DefaultProprecessor with Witvliet2020Preprocessor7, a more up-to-date connectome of C. elegans.
        # return Witvliet2020Preprocessor7().preprocess(save_as="graph_tensors.pt") # Note: Instantiation needed

        neurons_all = set(self.neuron_labels)
        sep = r"[\t,]"

        # --- Load Data ---
        try:
            GHermChem_Edges_df = pd.read_csv(
                os.path.join(RAW_DATA_DIR, "GHermChem_Edges.csv"),
                sep=sep,
                engine="python",
            )
            GHermElec_Sym_Edges_df = pd.read_csv(
                os.path.join(RAW_DATA_DIR, "GHermElec_Sym_Edges.csv"),
                sep=sep,
                engine="python",
            )
            GHermChem_Nodes_df = pd.read_csv(
                os.path.join(RAW_DATA_DIR, "GHermChem_Nodes.csv"),
                sep=sep,
                engine="python",
            )
            GHermElec_Sym_Nodes_df = pd.read_csv(
                os.path.join(RAW_DATA_DIR, "GHermElec_Sym_Nodes.csv"),
                sep=sep,
                engine="python",
            )
        except FileNotFoundError as e:
            print(
                f"Error loading default connectome CSV files: {e}. See class docstring for source info."
            )
            raise

        # --- Standardize and Filter Node Lists ---
        def standardize_and_filter_nodes(df_nodes):
            # Specific name cleaning for this dataset format
            df_nodes["Name"] = [
                v.replace("0", "") if not str(v).endswith("0") else str(v)
                for v in df_nodes["Name"]
            ]
            return df_nodes[df_nodes["Name"].isin(neurons_all)].reset_index(drop=True)

        Gsyn_nodes_df = standardize_and_filter_nodes(GHermChem_Nodes_df)
        Ggap_nodes_df = standardize_and_filter_nodes(GHermElec_Sym_Nodes_df)
        valid_syn_neurons = set(Gsyn_nodes_df.Name)
        valid_gap_neurons = set(Ggap_nodes_df.Name)

        chem_data = []  # List[Tuple[int, int, float]]
        gap_data = []  # List[Tuple[int, int, float]]

        # --- Process Chemical Synapses ---
        df_chem = GHermChem_Edges_df
        df_chem["EndNodes_1"] = [
            v.replace("0", "") if not str(v).endswith("0") else str(v)
            for v in df_chem["EndNodes_1"]
        ]
        df_chem["EndNodes_2"] = [
            v.replace("0", "") if not str(v).endswith("0") else str(v)
            for v in df_chem["EndNodes_2"]
        ]

        for _, row in df_chem.iterrows():
            n1_label, n2_label, weight = (
                row["EndNodes_1"],
                row["EndNodes_2"],
                row["Weight"],
            )
            if (
                n1_label in self.neuron_to_idx
                and n2_label in self.neuron_to_idx
                and n1_label in valid_syn_neurons
                and n2_label in valid_syn_neurons
            ):
                try:
                    weight_float = float(weight)
                    if weight_float != 0:
                        idx1 = self.neuron_to_idx[n1_label]
                        idx2 = self.neuron_to_idx[n2_label]
                        chem_data.append((idx1, idx2, weight_float))
                except (ValueError, TypeError):
                    pass  # Ignore non-numeric weights

        # --- Process Gap Junctions ---
        df_gap = GHermElec_Sym_Edges_df
        df_gap["EndNodes_1"] = [
            v.replace("0", "") if not str(v).endswith("0") else str(v)
            for v in df_gap["EndNodes_1"]
        ]
        df_gap["EndNodes_2"] = [
            v.replace("0", "") if not str(v).endswith("0") else str(v)
            for v in df_gap["EndNodes_2"]
        ]

        for _, row in df_gap.iterrows():
            n1_label, n2_label, weight = (
                row["EndNodes_1"],
                row["EndNodes_2"],
                row["Weight"],
            )
            if (
                n1_label in self.neuron_to_idx
                and n2_label in self.neuron_to_idx
                and n1_label in valid_gap_neurons
                and n2_label in valid_gap_neurons
            ):
                try:
                    weight_float = float(weight)
                    if weight_float != 0:
                        idx1 = self.neuron_to_idx[n1_label]
                        idx2 = self.neuron_to_idx[n2_label]
                        gap_data.append((idx1, idx2, weight_float))
                except (ValueError, TypeError):
                    pass  # Ignore non-numeric weights

        # --- Call Helpers to Process Edges ---
        edge_index, edge_attr = self._process_and_coalesce_edges(chem_data, gap_data)

        # --- Common Tasks & Save ---
        graph, node_type, node_label, node_index, node_class, num_classes = (
            self._preprocess_common_tasks(edge_index, edge_attr)
        )
        self._save_graph_tensors(
            save_as, graph, node_type, node_label, node_index, node_class, num_classes
        )


class ChklovskiiPreprocessor(ConnectomeBasePreprocessor):
    """
    Preprocessor for the Chklovskii connectome data.

    This class extends the ConnectomeBasePreprocessor to provide specific preprocessing
    steps for the Chklovskii connectome data from the 'NeuronConnect.csv' sheet.
    It includes methods for loading, processing, and saving the connectome data.

    Methods:
        preprocess(save_as="graph_tensors_chklkovskii.pt"):
            Preprocesses the Chklovskii connectome data and saves the graph tensors to a file.
    """

    def preprocess(self, save_as="graph_tensors_chklovskii.pt"):
        """
        Preprocesses the Chklovskii et al connectome data and saves the graph
        tensors to a file.

        Reads connectivity data from the 'NeuronConnect.csv' sheet, identifies
        chemical ('Sp') and gap junction ('EJ') types, collects raw edge tuples,
        calls base helpers to symmetrize gaps and coalesce edges, then processes
        common tasks and saves the final graph.

        The data is read from the XLS file named "Chklovskii_NeuronConnect.xls", which is a renaming of
        the file downloaded from https://www.wormatlas.org/images/NeuronConnect.xls. The connectome table
        is in the 'NeuronConnect.csv' sheet.

        NOTE: Description of this data from https://wormwiring.org/:
        Adult hermaphrodite, Data of Chen, Hall, and Chklovskii, 2006, Wiring optimization can relate neuronal structure and function, PNAS 103: 4723-4728 (doi:10.1073/pnas.0506806103)
        and Varshney, Chen, Paniaqua, Hall and Chklovskii, 2011, Structural properties of the C. elegans neuronal network, PLoS Comput. Biol. 3:7:e1001066 (doi:10.1371/journal.pcbi.1001066).
        Data of White et al., 1986, with additional connectivity in the ventral cord from reannotation of original electron micrographs.
        Connectivity table available through WormAtlas.org: Connectivity Data-download [.xls]
        Number of chemical and gap junction (electrical) synapses for all neurons and motor neurons. Number of NMJ’s for all ventral cord motor neurons.

        Args:
            save_as (str, optional): The name of the file to save the graph tensors to. Default is "graph_tensors_cklkovskii.pt".
        """
        # Load the XLS file and extract data from the 'NeuronConnect.csv' sheet
        try:
            df = pd.read_excel(
                os.path.join(RAW_DATA_DIR, "Chklovskii_NeuronConnect.xls"),
                sheet_name="NeuronConnect.csv",
            )
        except FileNotFoundError as e:
            print(f"ERROR: Chklovskii_NeuronConnect.xls not found in {RAW_DATA_DIR}")
            # Add source info from original docstring if desired
            raise e

        chem_data = [] # List[(neuron1_idx, neuron2_idx, weight)]
        gap_data = [] # List[(neuron1_idx, neuron2_idx, weight)]

        # Iterate over each row in the DataFrame
        for i in range(len(df)):
            neuron1_label = df.loc[i, "Neuron 1"]  # Pre-synaptic neuron
            neuron2_label = df.loc[i, "Neuron 2"]  # Post-synaptic neuron
            synapse_type = df.loc[i, "Type"]  # Synapse type (e.g., EJ, Sp)
            num_connections_raw = df.loc[i, "Nbr"]  # Number of connections

            if neuron1_label in self.neuron_labels and neuron2_label in self.neuron_labels:
                try:
                    weight = float(num_connections_raw)
                    if np.isnan(weight) or weight == 0:
                        continue # skip adding this edge since it's just 0
                    
                    idx1 = self.neuron_to_idx[neuron1_label]
                    idx2 = self.neuron_to_idx[neuron2_label]
                    
                    if synapse_type == "EJ":  # Electrical synapse (Gap Junction)
                        gap_data.append((idx1, idx2, weight))
                    elif synapse_type == "Sp":  # Only process "Sp" type chemical synapses
                        chem_data.append((idx1, idx2, weight))

                except (ValueError, TypeError):
                    # Ignore rows where 'Nbr' isn't a valid number
                    print(
                        f"Warning: Could not parse weight '{num_connections_raw}' for ({neuron1_label}, {neuron2_label}). Skipping."
                    )
                
        edge_index, edge_attr = self._process_and_coalesce_edges(chem_data, gap_data)
        
        # Perform common preprocessing tasks to create graph tensors
        graph, node_type, node_label, node_index, node_class, num_classes = (
            self._preprocess_common_tasks(edge_index, edge_attr)
        )

        # Save the processed graph tensors to the specified file
        self._save_graph_tensors(
            save_as,
            graph,
            node_type,
            node_label,
            node_index,
            node_class,
            num_classes,
        )


class OpenWormPreprocessor(ConnectomeBasePreprocessor):
    """
    Preprocessor for the OpenWorm connectome data.

    This class extends the ConnectomeBasePreprocessor to provide specific preprocessing
    steps for the OpenWorm connectome data. It includes methods for loading, processing,
    and saving the connectome data directly from the xls file.

    Methods:
        preprocess(save_as="graph_tensors_openworm.pt"):
            Preprocesses the OpenWorm connectome data and saves the graph tensors to a file.
    """

    def preprocess(self, save_as="graph_tensors_openworm.pt"):
        """
        Loads OpenWorm connectome from Excel, processes using base helpers, and saves tensors.

        Reads data from the 'Connectome' sheet, identifies chemical ('Send') and
        'GapJunction' types, collects raw edge tuples, calls base helpers to
        symmetrize gaps and coalesce edges, then processes common tasks and saves.

        Args:
            save_as (str, optional): Filename for saved tensors. Default "graph_tensors_openworm.pt".
        """
        try:
            df = pd.read_excel(
                os.path.join(RAW_DATA_DIR, "OpenWorm_CElegansNeuronTables.xls"),
                sheet_name="Connectome",
            )
        except FileNotFoundError:
            print(
                f"ERROR: OpenWorm_CElegansNeuronTables.xls not found in {RAW_DATA_DIR}"
            )
            raise
        except Exception as e:
            print(f"Error loading OpenWorm Excel file: {e}")
            raise

        chem_data = []  # List[Tuple[int, int, float]]
        gap_data = []  # List[Tuple[int, int, float]]

        # Iterate through rows and collect raw data tuples
        for i in range(len(df)):
            neuron1_label = df.loc[i, "Origin"]
            neuron2_label = df.loc[i, "Target"]
            synapse_type = df.loc[i, "Type"]
            num_connections_raw = df.loc[i, "Number of Connections"]

            if (
                neuron1_label in self.neuron_to_idx
                and neuron2_label in self.neuron_to_idx
            ):
                try:
                    weight = float(num_connections_raw)
                    if np.isnan(weight) or weight == 0:
                        continue  # Skip zero or invalid weights

                    idx1 = self.neuron_to_idx[neuron1_label]
                    idx2 = self.neuron_to_idx[neuron2_label]

                    if synapse_type == "GapJunction":
                        gap_data.append((idx1, idx2, weight))
                    elif synapse_type == "Send":  # 'Send' means chemical
                        chem_data.append((idx1, idx2, weight))

                except (ValueError, TypeError):
                    # Ignore rows where 'Number of Connections' isn't a valid number
                    pass

        # --- Call Helpers to Process Edges ---
        edge_index, edge_attr = self._process_and_coalesce_edges(chem_data, gap_data)

        # --- Common Tasks & Save ---
        graph, node_type, node_label, node_index, node_class, num_classes = (
            self._preprocess_common_tasks(edge_index, edge_attr)
        )
        self._save_graph_tensors(
            save_as, graph, node_type, node_label, node_index, node_class, num_classes
        )

class Randi2023Preprocessor(ConnectomeBasePreprocessor):
    """
    Preprocessor for the Randi et al., 2023 connectome data.

    This class extends the ConnectomeBasePreprocessor to provide specific preprocessing
    steps for the Randi et al., 2023 connectome data. It includes methods for loading,
    processing, and saving the connectome data.

    Methods:
        preprocess(save_as="graph_tensors_funconn.pt"):
            Preprocesses the Randi et al., 2023 connectome data and saves the graph tensors to a file.
    """

    def preprocess(self, save_as="graph_tensors_funconn.pt"):
        """
        Preprocesses the Randi et al., 2023 connectome data and saves the graph tensors to a file.

        The data is read from an Excel file named "CElegansFunctionalConnectivity.xlsx" which is a renaming of the
        Supplementary Table 1 file "1586_2023_6683_MOESM3_ESM.xlsx" downloaded from the Supplementary information of the paper
        "Randi, F., Sharma, A. K., Dvali, S., & Leifer, A. M. (2023). Neural signal propagation atlas of Caenorhabditis elegans. Nature, 623(7986), 406–414. https://doi.org/10.1038/s41586-023-06683-4"
        at this direct link: https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-023-06683-4/MediaObjects/41586_2023_6683_MOESM3_ESM.xlsx

        Args:
            save_as (str, optional): The name of the file to save the graph tensors to. Default is "graph_tensors_funconn.pt".

        Steps:
            1. Load the connectivity and significance data from "CElegansFunctionalConnectivity.xlsx".
            2. Initialize lists for edges and edge attributes.
            3. Iterate through the connectivity DataFrame:
                - Extract neuron pairs and their connectivity values.
                - Check significance and append edges and attributes to the respective lists.
            4. Convert edge attributes and edge indices to tensors.
            5. Call the `_preprocess_common_tasks` method to perform common preprocessing tasks.
            6. Save the graph tensors to the specified file.
        """
        edges = []
        edge_attr = []

        xls = pd.ExcelFile(os.path.join(RAW_DATA_DIR, "CElegansFunctionalConnectivity.xlsx"))
        df_connectivity = pd.read_excel(xls, sheet_name=0, index_col=0)
        df_significance = pd.read_excel(xls, sheet_name=1, index_col=0)

        for i, (row_label, row) in enumerate(df_connectivity.iterrows()):
            for j, (col_label, value) in enumerate(row.items()):
                if pd.isna(value) or np.isnan(value):
                    continue
                if row_label in self.neuron_labels and col_label in self.neuron_labels:
                    if df_significance.loc[row_label, col_label] < 0.05:
                        edges.append([row_label, col_label])
                        edge_attr.append([0, value])

        neuron_to_idx = {label: idx for idx, label in enumerate(self.neuron_labels)}
        edge_attr = torch.tensor(edge_attr)
        edge_index = torch.tensor(
            [[neuron_to_idx[neuron1], neuron_to_idx[neuron2]] for neuron1, neuron2 in edges]
        ).T

        # Perform common preprocessing tasks to create graph tensors
        graph, node_type, node_label, node_index, node_class, num_classes = (
            self._preprocess_common_tasks(edge_index, edge_attr)
        )

        # Save the processed graph tensors to the specified file
        self._save_graph_tensors(
            save_as,
            graph,
            node_type,
            node_label,
            node_index,
            node_class,
            num_classes,
        )


class Witvliet2020Preprocessor7(ConnectomeBasePreprocessor):
    """
    Preprocessor for the Witvliet et al., 2020 connectome data (adult 7).

    This class extends the ConnectomeBasePreprocessor to provide specific preprocessing
    steps for the Witvliet et al., 2020 connectome data for adult 7. It includes methods
    for loading, processing, and saving the connectome data.

    Methods:
        preprocess(save_as="graph_tensors_witvliet2020_7.pt"):
            Preprocesses the Witvliet et al., 2020 connectome data for adult 7 and saves the graph tensors to a file.
    """

    def preprocess(self, save_as="graph_tensors_witvliet2020_7.pt"):
        """
        Preprocesses the Witvliet et al., 2020 connectome data for adult 7 and saves the graph tensors to a file.

        The data is read from a CSV file named "witvliet_2020_7.csv".

        Args:
            save_as (str, optional): The name of the file to save the graph tensors to. Default is "graph_tensors_witvliet2020_7.pt".

        Steps:
            1. Load the connectome data from "witvliet_2020_7.csv".
            2. Initialize lists for edges and edge attributes.
            3. Iterate through the rows of the DataFrame:
                - Extract neuron pairs and synapse type.
                - Append edges and attributes to the respective lists.
                - Ensure symmetry for electrical synapses by adding reverse direction edges.
            4. Convert edge attributes and edge indices to tensors.
            5. Call the `_preprocess_common_tasks` method to perform common preprocessing tasks.
            6. Save the graph tensors to the specified file.
        """
        df = pd.read_csv(os.path.join(RAW_DATA_DIR, "witvliet_2020_7.csv"), sep=r"[\t,]")

        edges = []
        edge_attr = []

        for i in range(len(df)):
            neuron1 = df.loc[i, "pre"]
            neuron2 = df.loc[i, "post"]

            if neuron1 in self.neuron_labels and neuron2 in self.neuron_labels:
                # NOTE: This file lists both types of edges in the same file with only the "type" column to differentiate.
                # Therefore as we go through the lines, when see the [neuron_i, neuron_j] pair appearing a second time it is a different
                # type of synapse (chemical vs. electrical) than the one appearing previously (electrical vs chemical, respectively).
                # The first synapse with [neuron_i, neuron_j] pair encountered could be either electrical or chemical.
                edge_type = df.loc[i, "type"]
                num_connections = df.loc[i, "synapses"]
                edges.append([neuron1, neuron2])
                if edge_type == "electrical":
                    edge_attr.append([num_connections, 0])
                    # Adding the reverse direction to ensure symmetry of gap junctions
                    edges.append([neuron2, neuron1])
                    edge_attr.append([num_connections, 0])
                elif edge_type == "chemical":
                    # NOTE: Chemical synapses are asymmetric
                    edge_attr.append([0, num_connections])

        edge_attr = torch.tensor(edge_attr)
        edge_index = torch.tensor(
            [
                [self.neuron_to_idx[neuron1], self.neuron_to_idx[neuron2]]
                for neuron1, neuron2 in edges
            ]
        ).T

        # Perform common preprocessing tasks to create graph tensors
        graph, node_type, node_label, node_index, node_class, num_classes = (
            self._preprocess_common_tasks(edge_index, edge_attr)
        )

        # Save the processed graph tensors to the specified file
        self._save_graph_tensors(
            save_as,
            graph,
            node_type,
            node_label,
            node_index,
            node_class,
            num_classes,
        )


class Witvliet2020Preprocessor8(ConnectomeBasePreprocessor):
    """
    Preprocessor for the Witvliet et al., 2020 connectome data (adult 8).

    This class extends the ConnectomeBasePreprocessor to provide specific preprocessing
    steps for the Witvliet et al., 2020 connectome data for adult 8. It includes methods
    for loading, processing, and saving the connectome data.

    Methods
    -------
    preprocess(save_as="graph_tensors_witvliet2020_8.pt")
        Preprocesses the Witvliet et al., 2020 connectome data for adult 8 and saves the graph tensors to a file.
        The data is read from a CSV file named "witvliet_2020_8.csv".
    """

    def preprocess(self, save_as="graph_tensors_witvliet2020_8.pt"):
        """
        Preprocesses the Witvliet et al., 2020 connectome data for adult 8 and saves the graph tensors to a file.

        The data is read from a CSV file named "witvliet_2020_8.csv".

        Args:
            save_as (str, optional): The name of the file to save the graph tensors to. Default is "graph_tensors_witvliet2020_8.pt".

        Steps:
            1. Load the connectome data from "witvliet_2020_8.csv".
            2. Initialize lists for edges and edge attributes.
            3. Iterate through the rows of the DataFrame:
                - Extract neuron pairs and synapse type.
                - Append edges and attributes to the respective lists.
                - Ensure symmetry for electrical synapses by adding reverse direction edges.
            4. Convert edge attributes and edge indices to tensors.
            5. Call the `_preprocess_common_tasks` method to perform common preprocessing tasks.
            6. Save the graph tensors to the specified file.
        """
        df = pd.read_csv(os.path.join(RAW_DATA_DIR, "witvliet_2020_8.csv"), sep=r"[\t,]")

        edges = []
        edge_attr = []

        for i in range(len(df)):
            neuron1 = df.loc[i, "pre"]
            neuron2 = df.loc[i, "post"]

            if neuron1 in self.neuron_labels and neuron2 in self.neuron_labels:
                # NOTE: This file lists both types of edges in the same file with only the "type" column to differentiate.
                # Therefore as we go through the lines, when see the [neuron_i, neuron_j] pair appearing a second time it is a different
                # type of synapse (chemical vs. electrical) than the one appearing previously (electrical vs chemical, respectively).
                # The first synapse with [neuron_i, neuron_j] pair encountered could be either electrical or chemical.
                edge_type = df.loc[i, "type"]
                num_connections = df.loc[i, "synapses"]
                edges.append([neuron1, neuron2])
                if edge_type == "electrical":
                    edge_attr.append([num_connections, 0])
                    # Adding the reverse direction to ensure symmetry of gap junctions
                    edges.append([neuron2, neuron1])
                    edge_attr.append([num_connections, 0])
                elif edge_type == "chemical":
                    # NOTE: Chemical synapses are asymmetric
                    edge_attr.append([0, num_connections])

        edge_attr = torch.tensor(edge_attr)
        edge_index = torch.tensor(
            [
                [self.neuron_to_idx[neuron1], self.neuron_to_idx[neuron2]]
                for neuron1, neuron2 in edges
            ]
        ).T

        # Perform common preprocessing tasks to create graph tensors
        graph, node_type, node_label, node_index, node_class, num_classes = (
            self._preprocess_common_tasks(edge_index, edge_attr)
        )

        # Save the processed graph tensors to the specified file
        self._save_graph_tensors(
            save_as,
            graph,
            node_type,
            node_label,
            node_index,
            node_class,
            num_classes,
        )


class Cook2019Preprocessor(ConnectomeBasePreprocessor):
    """
    Preprocessor for the Cook et al., 2019 connectome data.

    This class extends the ConnectomeBasePreprocessor to provide specific preprocessing
    steps for the Cook et al., 2019 connectome data. It includes methods for loading,
    processing, and saving the connectome data.

    Methods:
        preprocess(save_as="graph_tensors_cook2019.pt"):
            Preprocesses the Cook et al., 2019 connectome data and saves the graph tensors to a file.
    """
    def preprocess(self, save_as="graph_tensors_cook2019.pt"):
        """
        Loads Cook et al. 2019 data from Excel, processes using base helpers, and saves tensors.

        Reads chemical and gap junction matrices from separate sheets, extracts raw
        edge tuples, calls base helpers to symmetrize gaps and coalesce edges,
        then processes common tasks and saves the final graph.

        Args:
            save_as (str, optional): Filename for saved tensors. Default "graph_tensors_cook2019.pt".
        """
        # --- Constants for sheet structure ---
        POST_SYNAPTIC_NEURON_ROW = 2
        PRE_SYNAPTIC_NEURON_COL = 2
        DATA_START_ROW = 3
        DATA_START_COL = 3
        chem_data = [] # List[Tuple[int, int, float]]
        gap_data = []  # List[Tuple[int, int, float]]

        # --- Load Excel File ---
        try:
            xlsx_file = pd.ExcelFile(os.path.join(RAW_DATA_DIR, "Cook2019.xlsx"))
        except FileNotFoundError:
            print(f"ERROR: Cook2019.xlsx not found in {RAW_DATA_DIR}")
            raise

        # --- Internal helper to extract raw data from a sheet matrix ---
        def extract_raw_data_from_sheet(df, data_list_to_append):
            # Extract neuron lists from the specific locations in this sheet format
            post_labels = df.iloc[POST_SYNAPTIC_NEURON_ROW, DATA_START_COL:].tolist()
            pre_labels = df.iloc[DATA_START_ROW:, PRE_SYNAPTIC_NEURON_COL].tolist()

            for j, pre_label in enumerate(pre_labels):
                # Skip if pre-neuron is invalid or not in our master list
                if pd.isna(pre_label) or pre_label not in self.neuron_to_idx: continue
                idx_pre = self.neuron_to_idx[pre_label]
                actual_row_index = DATA_START_ROW + j

                for i, post_label in enumerate(post_labels):
                    # Skip if post-neuron is invalid or not in our master list
                    if pd.isna(post_label) or post_label not in self.neuron_to_idx: continue
                    idx_post = self.neuron_to_idx[post_label]
                    actual_col_index = DATA_START_COL + i

                    try:
                        weight = df.iloc[actual_row_index, actual_col_index]
                        if not pd.isna(weight):
                            weight_float = float(weight)
                            if weight_float != 0:
                                data_list_to_append.append((idx_pre, idx_post, weight_float))
                    except (IndexError, ValueError, TypeError):
                        pass # Ignore errors during cell processing

        # --- Process Sheets ---
        try:
            df_chem = pd.read_excel(xlsx_file, sheet_name="hermaphrodite chemical", header=None)
            extract_raw_data_from_sheet(df_chem, chem_data)
        except Exception as e:
            print(f"Error processing Cook2019 chemical sheet: {e}")
            raise(e)

        try:
            df_gap = pd.read_excel(xlsx_file, sheet_name="hermaphrodite gap jn symmetric", header=None)
            extract_raw_data_from_sheet(df_gap, gap_data)
        except Exception as e:
            print(f"Error processing Cook2019 gap junction sheet: {e}")
            raise(e)

        # --- Call Helper to Process Edges ---
        edge_index, edge_attr = self._process_and_coalesce_edges(chem_data, gap_data)

        # --- Common Tasks & Save ---
        graph, node_type, node_label, node_index, node_class, num_classes = (
            self._preprocess_common_tasks(edge_index, edge_attr)
        )
        self._save_graph_tensors(
            save_as, graph, node_type, node_label, node_index, node_class, num_classes
        )


class White1986WholePreprocessor(ConnectomeBasePreprocessor):
    """
    Preprocessor for the White et al., 1986 connectome data (whole).

    This class extends the ConnectomeBasePreprocessor to provide specific preprocessing
    steps for the White et al., 1986 connectome data for the whole organism. It includes methods
    for loading, processing, and saving the connectome data.

    Methods:
        preprocess(save_as="graph_tensors_white1986_whole.pt"):
            Preprocesses the White et al., 1986 connectome data for the whole organism and saves the graph tensors to a file.
    """

    def preprocess(self, save_as="graph_tensors_white1986_whole.pt"):
        """
        Preprocesses the White et al., 1986 connectome data for the whole organism and saves the graph tensors to a file.

        The data is read from a CSV file named "white_1986_whole.csv".

        Args:
            save_as (str, optional): The name of the file to save the graph tensors to. Default is "graph_tensors_white1986_whole.pt".

        Steps:
            1. Load the connectome data from "white_1986_whole.csv".
            2. Initialize lists for edges and edge attributes.
            3. Iterate through the rows of the DataFrame:
                - Extract neuron pairs and synapse type.
                - Append edges and attributes to the respective lists.
                - Ensure symmetry for electrical synapses by adding reverse direction edges.
            4. Convert edge attributes and edge indices to tensors.
            5. Call the `_preprocess_common_tasks` method to perform common preprocessing tasks.
            6. Save the graph tensors to the specified file.
        """
        df = pd.read_csv(os.path.join(RAW_DATA_DIR, "white_1986_whole.csv"), sep=r"[\t,]")

        edges = []
        edge_attr = []

        for i in range(len(df)):
            neuron1 = df.loc[i, "pre"]
            neuron2 = df.loc[i, "post"]

            if neuron1 in self.neuron_labels and neuron2 in self.neuron_labels:
                # NOTE: This file lists both types of edges in the same file with only the "type" column to differentiate.
                # Therefore as we go through the lines, when see the [neuron_i, neuron_j] pair appearing a second time it is a different
                # type of synapse (chemical vs. electrical) than the one appearing previously (electrical vs chemical, respectively).
                # The first synapse with [neuron_i, neuron_j] pair encountered could be either electrical or chemical.
                edge_type = df.loc[i, "type"]
                num_connections = df.loc[i, "synapses"]
                edges.append([neuron1, neuron2])
                if edge_type == "electrical":
                    edge_attr.append([num_connections, 0])
                    # Adding the reverse direction to ensure symmetry of gap junctions
                    edges.append([neuron2, neuron1])
                    edge_attr.append([num_connections, 0])
                elif edge_type == "chemical":
                    # NOTE: Chemical synapses are asymmetric
                    edge_attr.append([0, num_connections])

        edge_attr = torch.tensor(edge_attr)
        edge_index = torch.tensor(
            [
                [self.neuron_to_idx[neuron1], self.neuron_to_idx[neuron2]]
                for neuron1, neuron2 in edges
            ]
        ).T

        # Perform common preprocessing tasks to create graph tensors
        graph, node_type, node_label, node_index, node_class, num_classes = (
            self._preprocess_common_tasks(edge_index, edge_attr)
        )

        # Save the processed graph tensors to the specified file
        self._save_graph_tensors(
            save_as,
            graph,
            node_type,
            node_label,
            node_index,
            node_class,
            num_classes,
        )


class White1986N2UPreprocessor(ConnectomeBasePreprocessor):
    """
    Preprocessor for the White et al., 1986 connectome data (N2U).

    This class extends the ConnectomeBasePreprocessor to provide specific preprocessing
    steps for the White et al., 1986 connectome data for the N2U organism. It includes methods
    for loading, processing, and saving the connectome data.

    Methods:
        preprocess(save_as="graph_tensors_white1986_n2u.pt"):
            Preprocesses the White et al., 1986 connectome data for the N2U organism and saves the graph tensors to a file.
    """

    def preprocess(self, save_as="graph_tensors_white1986_n2u.pt"):
        """
        Preprocesses the White et al., 1986 connectome data for the N2U organism and saves the graph tensors to a file.

        The data is read from a CSV file named "white_1986_n2u.csv".

        Args:
            save_as (str, optional): The name of the file to save the graph tensors to. Default is "graph_tensors_white1986_n2u.pt".

        Steps:
            1. Load the connectome data from "white_1986_n2u.csv".
            2. Initialize lists for edges and edge attributes.
            3. Iterate through the rows of the DataFrame:
                - Extract neuron pairs and synapse type.
                - Append edges and attributes to the respective lists.
                - Ensure symmetry for electrical synapses by adding reverse direction edges.
            4. Convert edge attributes and edge indices to tensors.
            5. Call the `_preprocess_common_tasks` method to perform common preprocessing tasks.
            6. Save the graph tensors to the specified file.
        """
        df = pd.read_csv(os.path.join(RAW_DATA_DIR, "white_1986_n2u.csv"), sep=r"[\t,]")

        edges = []
        edge_attr = []

        for i in range(len(df)):
            neuron1 = df.loc[i, "pre"]
            neuron2 = df.loc[i, "post"]

            if neuron1 in self.neuron_labels and neuron2 in self.neuron_labels:
                # NOTE: This file lists both types of edges in the same file with only the "type" column to differentiate.
                # Therefore as we go through the lines, when see the [neuron_i, neuron_j] pair appearing a second time it is a different
                # type of synapse (chemical vs. electrical) than the one appearing previously (electrical vs chemical, respectively).
                # The first synapse with [neuron_i, neuron_j] pair encountered could be either electrical or chemical.
                edge_type = df.loc[i, "type"]
                num_connections = df.loc[i, "synapses"]
                edges.append([neuron1, neuron2])
                if edge_type == "electrical":
                    edge_attr.append([num_connections, 0])
                    # Adding the reverse direction to ensure symmetry of gap junctions
                    edges.append([neuron2, neuron1])
                    edge_attr.append([num_connections, 0])
                elif edge_type == "chemical":
                    # NOTE: Chemical synapses are asymmetric
                    edge_attr.append([0, num_connections])

        edge_attr = torch.tensor(edge_attr)
        edge_index = torch.tensor(
            [
                [self.neuron_to_idx[neuron1], self.neuron_to_idx[neuron2]]
                for neuron1, neuron2 in edges
            ]
        ).T

        # Perform common preprocessing tasks to create graph tensors
        graph, node_type, node_label, node_index, node_class, num_classes = (
            self._preprocess_common_tasks(edge_index, edge_attr)
        )

        # Save the processed graph tensors to the specified file
        self._save_graph_tensors(
            save_as,
            graph,
            node_type,
            node_label,
            node_index,
            node_class,
            num_classes,
        )


class White1986JSHPreprocessor(ConnectomeBasePreprocessor):
    """
    Preprocessor for the White et al., 1986 connectome data (JSH).

    This class extends the ConnectomeBasePreprocessor to provide specific preprocessing
    steps for the White et al., 1986 connectome data for the JSH organism. It includes methods
    for loading, processing, and saving the connectome data.

    Methods:
        preprocess(save_as="graph_tensors_white1986_jsh.pt"):
            Preprocesses the White et al., 1986 connectome data for the JSH organism and saves the graph tensors to a file.
    """

    def preprocess(self, save_as="graph_tensors_white1986_jsh.pt"):
        """
        Preprocesses the White et al., 1986 connectome data for the JSH organism and saves the graph tensors to a file.

        The data is read from a CSV file named "white_1986_jsh.csv".

        Args:
            save_as (str, optional): The name of the file to save the graph tensors to. Default is "graph_tensors_white1986_jsh.pt".

        Steps:
            1. Load the connectome data from "white_1986_jsh.csv".
            2. Initialize lists for edges and edge attributes.
            3. Iterate through the rows of the DataFrame:
                - Extract neuron pairs and synapse type.
                - Append edges and attributes to the respective lists.
                - Ensure symmetry for electrical synapses by adding reverse direction edges.
            4. Convert edge attributes and edge indices to tensors.
            5. Call the `_preprocess_common_tasks` method to perform common preprocessing tasks.
            6. Save the graph tensors to the specified file.
        """
        df = pd.read_csv(os.path.join(RAW_DATA_DIR, "white_1986_jsh.csv"), sep=r"[\t,]")

        edges = []
        edge_attr = []

        for i in range(len(df)):
            neuron1 = df.loc[i, "pre"]
            neuron2 = df.loc[i, "post"]

            if neuron1 in self.neuron_labels and neuron2 in self.neuron_labels:
                # NOTE: This file lists both types of edges in the same file with only the "type" column to differentiate.
                # Therefore as we go through the lines, when see the [neuron_i, neuron_j] pair appearing a second time it is a different
                # type of synapse (chemical vs. electrical) than the one appearing previously (electrical vs chemical, respectively).
                # The first synapse with [neuron_i, neuron_j] pair encountered could be either electrical or chemical.
                edge_type = df.loc[i, "type"]
                num_connections = df.loc[i, "synapses"]
                edges.append([neuron1, neuron2])
                if edge_type == "electrical":
                    edge_attr.append([num_connections, 0])
                    # Adding the reverse direction to ensure symmetry of gap junctions
                    edges.append([neuron2, neuron1])
                    edge_attr.append([num_connections, 0])
                elif edge_type == "chemical":
                    # NOTE: Chemical synapses are asymmetric
                    edge_attr.append([0, num_connections])

        edge_attr = torch.tensor(edge_attr)
        edge_index = torch.tensor(
            [
                [self.neuron_to_idx[neuron1], self.neuron_to_idx[neuron2]]
                for neuron1, neuron2 in edges
            ]
        ).T

        # Perform common preprocessing tasks to create graph tensors
        graph, node_type, node_label, node_index, node_class, num_classes = (
            self._preprocess_common_tasks(edge_index, edge_attr)
        )

        # Save the processed graph tensors to the specified file
        self._save_graph_tensors(
            save_as,
            graph,
            node_type,
            node_label,
            node_index,
            node_class,
            num_classes,
        )


class White1986JSEPreprocessor(ConnectomeBasePreprocessor):
    """
    Preprocessor for the White et al., 1986 connectome data (JSE).

    This class extends the ConnectomeBasePreprocessor to provide specific preprocessing
    steps for the White et al., 1986 connectome data for the JSE organism. It includes methods
    for loading, processing, and saving the connectome data.

    Methods:
        preprocess(save_as="graph_tensors_white1986_jse.pt"):
            Preprocesses the White et al., 1986 connectome data for the JSE organism and saves the graph tensors to a file.
    """

    def preprocess(self, save_as="graph_tensors_white1986_jse.pt"):
        """
        Preprocesses the White et al., 1986 connectome data for the JSE organism and saves the graph tensors to a file.

        The data is read from a CSV file named "white_1986_jse.csv".

        Args:
            save_as (str, optional): The name of the file to save the graph tensors to. Default is "graph_tensors_white1986_jse.pt".

        Steps:
            1. Load the connectome data from "white_1986_jse.csv".
            2. Initialize lists for edges and edge attributes.
            3. Iterate through the rows of the DataFrame:
                - Extract neuron pairs and synapse type.
                - Append edges and attributes to the respective lists.
                - Ensure symmetry for electrical synapses by adding reverse direction edges.
            4. Convert edge attributes and edge indices to tensors.
            5. Call the `_preprocess_common_tasks` method to perform common preprocessing tasks.
            6. Save the graph tensors to the specified file.
        """
        df = pd.read_csv(os.path.join(RAW_DATA_DIR, "white_1986_jse.csv"), sep=r"[\t,]")

        edges = []
        edge_attr = []

        for i in range(len(df)):
            neuron1 = df.loc[i, "pre"]
            neuron2 = df.loc[i, "post"]

            if neuron1 in self.neuron_labels and neuron2 in self.neuron_labels:
                # NOTE: This file lists both types of edges in the same file with only the "type" column to differentiate.
                # Therefore as we go through the lines, when see the [neuron_i, neuron_j] pair appearing a second time it is a different
                # type of synapse (chemical vs. electrical) than the one appearing previously (electrical vs chemical, respectively).
                # The first synapse with [neuron_i, neuron_j] pair encountered could be either electrical or chemical.
                edge_type = df.loc[i, "type"]
                num_connections = df.loc[i, "synapses"]
                edges.append([neuron1, neuron2])
                if edge_type == "electrical":
                    edge_attr.append([num_connections, 0])
                    # Adding the reverse direction to ensure symmetry of gap junctions
                    edges.append([neuron2, neuron1])
                    edge_attr.append([num_connections, 0])
                elif edge_type == "chemical":
                    # NOTE: Chemical synapses are asymmetric
                    edge_attr.append([0, num_connections])

        edge_attr = torch.tensor(edge_attr)
        edge_index = torch.tensor(
            [
                [self.neuron_to_idx[neuron1], self.neuron_to_idx[neuron2]]
                for neuron1, neuron2 in edges
            ]
        ).T

        # Perform common preprocessing tasks to create graph tensors
        graph, node_type, node_label, node_index, node_class, num_classes = (
            self._preprocess_common_tasks(edge_index, edge_attr)
        )

        # Save the processed graph tensors to the specified file
        self._save_graph_tensors(
            save_as,
            graph,
            node_type,
            node_label,
            node_index,
            node_class,
            num_classes,
        )
