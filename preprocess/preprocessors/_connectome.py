"""
Contains dataset-specific connectome preprocessors,
which are subclasses of ConnectomeBasePreprocessor.

Count: 10
"""
from preprocess._pkg import *
from preprocess.preprocessors._helpers import *
from preprocess.preprocessors._base_preprocessors import ConnectomeBasePreprocessor, DefaultPreprocessor


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
        Preprocesses the Chklovskii et al connectome data and saves the graph tensors to a file.

        The data is read from the XLS file named "Chklovskii_NeuronConnect.xls", which is a renaming of
        the file downloaded from https://www.wormatlas.org/images/NeuronConnect.xls. The connectome table
        is in the 'NeuronConnect.csv' sheet.

        NOTE: Description of this data from https://wormwiring.org/:
        Adult hermaphrodite, Data of Chen, Hall, and Chklovskii, 2006, Wiring optimization can relate neuronal structure and function, PNAS 103: 4723-4728 (doi:10.1073/pnas.0506806103)
        and Varshney, Chen, Paniaqua, Hall and Chklovskii, 2011, Structural properties of the C. elegans neuronal network, PLoS Comput. Biol. 3:7:e1001066 (doi:10.1371/journal.pcbi.1001066).
        Data of White et al., 1986, with additional connectivity in the ventral cord from reannotation of original electron micrographs.
        Connectivity table available through WormAtlas.org: Connectivity Data-download [.xls]
        Number of chemical and gap junction (electrical) synapses for all neurons and motor neurons. Number of NMJ’s for all ventral cord motor neurons.

        For chemical synapses, only entries with type "Sp" (send reannotated) are considered to
        avoid redundant connections.

        Args:
            save_as (str, optional): The name of the file to save the graph tensors to. Default is "graph_tensors_cklkovskii.pt".

        Steps:
            1. Load the connectome data from the 'NeuronConnect.csv' sheet in "Chklovskii_NeuronConnect.xls".
            2. Only consider rows with "Sp" (chemical) and "EJ" (gap junction) types.
            3. Append edges and attributes (synapse strength).
            4. Ensure symmetry for electrical synapses.
            5. Convert edge attributes and edge indices to tensors.
            6. Call the `preprocess_common_tasks` method to create graph tensors.
            7. Save the graph tensors to the specified file.
        """
        # Load the XLS file and extract data from the 'NeuronConnect.csv' sheet
        df = pd.read_excel(
            os.path.join(RAW_DATA_DIR, "Chklovskii_NeuronConnect.xls"),
            sheet_name="NeuronConnect.csv",
        )

        edges = []
        edge_attr = []

        # Iterate over each row in the DataFrame
        for i in range(len(df)):
            neuron1 = df.loc[i, "Neuron 1"]  # Pre-synaptic neuron
            neuron2 = df.loc[i, "Neuron 2"]  # Post-synaptic neuron
            synapse_type = df.loc[i, "Type"]  # Synapse type (e.g., EJ, Sp)
            num_connections = df.loc[i, "Nbr"]  # Number of connections

            if neuron1 in self.neuron_labels and neuron2 in self.neuron_labels:
                if synapse_type == "EJ":  # Electrical synapse (Gap Junction)
                    edges.append([neuron1, neuron2])
                    edge_attr.append(
                        [num_connections, 0]
                    )  # Electrical synapse encoded as [num_connections, 0]

                    # Ensure symmetry by adding reverse direction for electrical synapses
                    edges.append([neuron2, neuron1])
                    edge_attr.append([num_connections, 0])

                elif synapse_type == "Sp":  # Only process "Sp" type chemical synapses
                    edges.append([neuron1, neuron2])
                    edge_attr.append(
                        [0, num_connections]
                    )  # Chemical synapse encoded as [0, num_connections]

        # Convert edge attributes and edge indices to torch tensors
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        edge_index = torch.tensor(
            [
                [self.neuron_to_idx[neuron1], self.neuron_to_idx[neuron2]]
                for neuron1, neuron2 in edges
            ]
        ).T

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
        Preprocesses the OpenWorm connectome data and saves the graph tensors to a file.

        The data is read directly from an XLS file named "OpenWorm_CElegansNeuronTables.xls", which is a rename of the
        file downloaded from the OpenWorm repository: https://github.com/openworm/c302/blob/master/c302/CElegansNeuronTables.xls

        Args:
            save_as (str, optional): The name of the file to save the graph tensors to. Default is "graph_tensors_openworm.pt".

        Steps:
            1. Load the connectome data from the first sheet of the "OpenWorm_CElegansNeuronTables.xls" file.
            2. Initialize lists for edges and edge attributes.
            3. Iterate through the rows of the DataFrame:
                - Extract neuron pairs and synapse type.
                - Append edges and attributes to the respective lists.
                - Ensure symmetry for electrical synapses by adding reverse direction edges.
            4. Convert edge attributes and edge indices to tensors.
            5. Call the `preprocess_common_tasks` method to perform common preprocessing tasks.
            6. Save the graph tensors to the specified file.
        """
        # Load the XLS file and extract data from the first sheet "Connectome"
        df = pd.read_excel(
            os.path.join(RAW_DATA_DIR, "OpenWorm_CElegansNeuronTables.xls"), sheet_name="Connectome"
        )

        edges = []
        edge_attr = []

        for i in range(len(df)):
            neuron1 = df.loc[i, "Origin"]
            neuron2 = df.loc[i, "Target"]

            if neuron1 in self.neuron_labels and neuron2 in self.neuron_labels:
                # Determine the synapse type and number of connections
                synapse_type = df.loc[i, "Type"]
                num_connections = df.loc[i, "Number of Connections"]

                # Add the connection between neuron1 and neuron2
                edges.append([neuron1, neuron2])

                if synapse_type == "GapJunction":  # electrical synapse
                    edge_attr.append(
                        [num_connections, 0]
                    )  # electrical synapse encoded as [num_connections, 0]

                    # Ensure symmetry for gap junctions by adding reverse connection
                    edges.append([neuron2, neuron1])
                    edge_attr.append([num_connections, 0])
                elif synapse_type == "Send":  # chemical synapse
                    edge_attr.append(
                        [0, num_connections]
                    )  # chemical synapse encoded as [0, num_connections]

        # Convert to torch tensors
        edge_attr = torch.tensor(edge_attr)
        edge_index = torch.tensor(
            [
                [self.neuron_to_idx[neuron1], self.neuron_to_idx[neuron2]]
                for neuron1, neuron2 in edges
            ]
        ).T

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
            5. Call the `preprocess_common_tasks` method to perform common preprocessing tasks.
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
            5. Call the `preprocess_common_tasks` method to perform common preprocessing tasks.
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
            5. Call the `preprocess_common_tasks` method to perform common preprocessing tasks.
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
        Preprocesses the Cook et al., 2019 connectome data and saves the graph tensors to a file.

        The data is read from an Excel file named "Cook2019.xlsx".

        Args:
            save_as (str, optional): The name of the file to save the graph tensors to. Default is "graph_tensors_cook2019.pt".

        Steps:
            1. Load the chemical synapse data from the "hermaphrodite chemical" sheet in "Cook2019.xlsx".
            2. Load the electrical synapse data from the "hermaphrodite gap jn symmetric" sheet in "Cook2019.xlsx".
            3. Initialize lists for edges and edge attributes.
            4. Iterate through the chemical synapse data:
                - Extract neuron pairs and their weights.
                - Append edges and attributes to the respective lists.
            5. Iterate through the electrical synapse data:
                - Extract neuron pairs and their weights.
                - Append edges and attributes to the respective lists.
                - Ensure symmetry for electrical synapses by adding reverse direction edges.
            6. Convert edge attributes and edge indices to tensors.
            7. Call the `preprocess_common_tasks` method to perform common preprocessing tasks.
            8. Save the graph tensors to the specified file.
        """
        edges = []
        edge_attr = []

        xlsx_file = pd.ExcelFile(os.path.join(RAW_DATA_DIR, "Cook2019.xlsx"))

        df = pd.read_excel(xlsx_file, sheet_name="hermaphrodite chemical")

        for i, line in enumerate(df):
            if i > 2:
                col_data = df.iloc[:-1, i]
                for j, weight in enumerate(col_data):
                    if j > 1 and not pd.isna(df.iloc[j, i]):
                        post = df.iloc[1, i]
                        pre = df.iloc[j, 2]
                        if pre in self.neuron_labels and post in self.neuron_labels:
                            edges.append([pre, post])
                            edge_attr.append(
                                [0, df.iloc[j, i]]
                            )  # second edge_attr feature is for gap junction weights

        df = pd.read_excel(xlsx_file, sheet_name="hermaphrodite gap jn symmetric")

        for i, line in enumerate(df):
            if i > 2:
                col_data = df.iloc[:-1, i]
                for j, weight in enumerate(col_data):
                    if j > 1 and not pd.isna(df.iloc[j, i]):
                        post = df.iloc[1, i]
                        pre = df.iloc[j, 2]
                        if pre in self.neuron_labels and post in self.neuron_labels:
                            if [pre, post] in edges:
                                edge_idx = edges.index([pre, post])
                                edge_attr[edge_idx][0] = df.iloc[
                                    j, i
                                ]  # first edge_attr feature is for gap junction weights
                            else:
                                edges.append([pre, post])
                                edge_attr.append([df.iloc[j, i], 0])

        edge_attr = torch.tensor(edge_attr)
        edge_index = torch.tensor(
            [
                [self.neuron_to_idx[neuron1], self.neuron_to_idx[neuron2]]
                for neuron1, neuron2 in edges
            ]
        ).T

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
            5. Call the `preprocess_common_tasks` method to perform common preprocessing tasks.
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
            5. Call the `preprocess_common_tasks` method to perform common preprocessing tasks.
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
            5. Call the `preprocess_common_tasks` method to perform common preprocessing tasks.
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
            5. Call the `preprocess_common_tasks` method to perform common preprocessing tasks.
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
