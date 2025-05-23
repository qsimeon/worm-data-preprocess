import os
import torch
import random
import logging
import warnings
import numpy as np
import pandas as pd
import torch.multiprocessing

# Configure logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Ignore all warnings
warnings.filterwarnings(action="ignore")

# Set the start method for multiprocessing
torch.multiprocessing.set_start_method("spawn", force=True)

# Set essential environment variables
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

# Some global variables
WORLD_SIZE = torch.cuda.device_count()
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(ROOT_DIR, "data", "raw")

RAW_DATA_URL = "https://www.dropbox.com/scl/fi/mp6qfxhf9h816y5apj9b1/raw_data.zip?rlkey=5tu0zkrge8atwccxjec0umkfb&dl=1"

LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# List of all 300 hermaphrodite neuron names
labels_file = os.path.join(RAW_DATA_DIR, "neuron_labels.txt")

# Master sheet of info on all 300 hermaphrodite neurons
neurons_master_file = os.path.join(RAW_DATA_DIR, "neuron_master_sheet.csv")

# if os.path.exists(labels_file):
if os.path.exists(neurons_master_file):

    # Load the labels from the master csv file
    print(f"Loading from {neurons_master_file}.\n")
    df_neurons_master = pd.read_csv(neurons_master_file).sort_values(by="label", ascending=True)
    NEURON_LABELS = df_neurons_master["label"].tolist()
elif os.path.exists(labels_file):
    # Load the labels from the text file
    NEURON_LABELS = sorted(
        pd.read_csv(
            labels_file,
            sep=" ",
            header=None,
            names=["neuron"],
        ).neuron
    )
else:
    # Enumarte all the labels manually (once) and save them to a text file
    # print(f"Saving to {labels_file}.\n")
    # fmt: off
    NEURON_LABELS = [ 
            # References: (1) https://www.wormatlas.org/neurons/Individual%20Neurons/Neuronframeset.html 
            #             (2) https://www.wormatlas.org/NeuronNames.htm
            "ADAL", "ADAR", "ADEL", "ADER", "ADFL", "ADFR", "ADLL", "ADLR", "AFDL", "AFDR",
            "AIAL", "AIAR", "AIBL", "AIBR", "AIML", "AIMR", "AINL", "AINR", "AIYL", "AIYR",
            "AIZL", "AIZR", "ALA", "ALML", "ALMR", "ALNL", "ALNR", "AQR", "AS1", "AS10",
            "AS11", "AS2", "AS3", "AS4", "AS5", "AS6", "AS7", "AS8", "AS9", "ASEL", "ASER",
            "ASGL", "ASGR", "ASHL", "ASHR", "ASIL", "ASIR", "ASJL", "ASJR", "ASKL", "ASKR",
            "AUAL", "AUAR", "AVAL", "AVAR", "AVBL", "AVBR", "AVDL", "AVDR", "AVEL", "AVER",
            "AVFL", "AVFR", "AVG", "AVHL", "AVHR", "AVJL", "AVJR", "AVKL", "AVKR", "AVL",
            "AVM", "AWAL", "AWAR", "AWBL", "AWBR", "AWCL", "AWCR", "BAGL", "BAGR", "BDUL",
            "BDUR", "CEPDL", "CEPDR", "CEPVL", "CEPVR", "DA1", "DA2", "DA3",
            "DA4", "DA5", "DA6", "DA7", "DA8", "DA9", "DB1", "DB2", "DB3", "DB4", "DB5",
            "DB6", "DB7", "DD1", "DD2", "DD3", "DD4", "DD5", "DD6", "DVA", "DVB", "DVC",
            "FLPL", "FLPR", "HSNL", "HSNR", "I1L", "I1R", "I2L", "I2R", "I3", "I4", "I5",
            "I6", "IL1DL", "IL1DR", "IL1L", "IL1R", "IL1VL", "IL1VR", "IL2DL", "IL2DR", "IL2L",
            "IL2R", "IL2VL", "IL2VR", "LUAL", "LUAR", "M1", "M2L", "M2R", "M3L", "M3R", "M4",
            "M5", "MCL", "MCR", "MI", "NSML", "NSMR", "OLLL", "OLLR", "OLQDL", "OLQDR",
            "OLQVL", "OLQVR", "PDA", "PDB", "PDEL", "PDER", "PHAL", "PHAR", "PHBL", "PHBR",
            "PHCL", "PHCR", "PLML", "PLMR", "PLNL", "PLNR", "PQR", "PVCL", "PVCR", "PVDL",
            "PVDR", "PVM", "PVNL", "PVNR", "PVPL", "PVPR", "PVQL", "PVQR", "PVR", "PVT",
            "PVWL", "PVWR", "RIAL", "RIAR", "RIBL", "RIBR", "RICL", "RICR", "RID", "RIFL",
            "RIFR", "RIGL", "RIGR", "RIH", "RIML", "RIMR", "RIPL", "RIPR", "RIR", "RIS",
            "RIVL", "RIVR", "RMDDL", "RMDDR", "RMDL", "RMDR", "RMDVL", "RMDVR", "RMED",
            "RMEL", "RMER", "RMEV", "RMFL", "RMFR", "RMGL", "RMGR", "RMHL", "RMHR", "SAADL",
            "SAADR", "SAAVL", "SAAVR", "SABD", "SABVL", "SABVR", "SDQL", "SDQR", "SIADL",
            "SIADR", "SIAVL", "SIAVR", "SIBDL", "SIBDR", "SIBVL", "SIBVR", "SMBDL", "SMBDR",
            "SMBVL", "SMBVR", "SMDDL", "SMDDR", "SMDVL", "SMDVR", "URADL", "URADR", "URAVL",
            "URAVR", "URBL", "URBR", "URXL", "URXR", "URYDL", "URYDR", "URYVL", "URYVR",
            "VA1", "VA10", "VA11", "VA12", "VA2", "VA3", "VA4", "VA5", "VA6", "VA7", "VA8",
            "VA9", "VB1", "VB10", "VB11", "VB2", "VB3", "VB4", "VB5", "VB6", "VB7", "VB8",
            "VB9", "VC1", "VC2", "VC3", "VC4", "VC5", "VC6", "VD1", "VD10", "VD11", "VD12",
            "VD13", "VD2", "VD3", "VD4", "VD5", "VD6", "VD7", "VD8", "VD9"
        ] # "CANL", "CANR"
    # NOTE: As of Cook et al. (2019), the CANL and CANR are considered to be end-organs, not neurons.
    # fmt: on

    # Write to a text file called "neuron_labels.txt" using pandas
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    pd.DataFrame(NEURON_LABELS).to_csv(labels_file, sep=" ", header=False, index=False)

NUM_NEURONS = len(NEURON_LABELS)
RAW_ZIP = "raw_data.zip"
RAW_FILES = [
    ### >>> Default (Premaratne's preprocessed Cook2019) connectome data files >>>
    "GHermChem_Edges.csv",
    "GHermChem_Nodes.csv",
    "GHermElec_Sym_Edges.csv",
    "GHermElec_Sym_Nodes.csv",
    ### <<< Default (Kamal Premaratne's preprocessed Cook2019) connectome data files <<<
    "Cook2019.xlsx",  # original Cook et al. (2019) connectome data file
    "Chklovskii_NeuronConnect.xls",  # Chen et al. (2006) & Varshney et al. (2011) connectome data file
    "OpenWorm_CElegansNeuronTables.xls",  # OpenWorm connectome data file (an earlier connectome (which one?) augmented with neurotransmitter type)
    "CElegansFunctionalConnectivity.xlsx",  # Randi et al. (2023) functional connectome data file
    "white_1986_jsh.csv",  # L4 brain
    "white_1986_n2u.csv",  # adult brain
    "white_1986_jse.csv",  # adult tail
    "white_1986_whole.csv",  # whole animal compilation of Varshey et al. (2011)
    "witvliet_2020_7.csv",  # one adult brain of Witvliet et al. (2020)
    "witvliet_2020_8.csv",  # another adult brain of Witvliet et al. (2020)
    "LowResAtlasWithHighResHeadsAndTails.csv",  # atlas of C. elegans neuron 3D positions
    "Hobert2016_BrainAtlas.xlsx",  # mapping between class, label, type, and neurotransmitter
    "Witvliet2020_NeuronClasses.xlsx",  # high level classes of hermaphrodite neurons
    "neuron_labels.txt",  # labels (names) of all hermaphrodite neurons
    "neuron_master_sheet.csv",  # TODO: working on a sheet combines all neuron information (labels, classes, neurotransmitter, cell type, and position)
    # NOTE: "neuron_master_sheet.csv", once complete, should make the following files obsolete: "neuron_labels.txt", ""Witvliet2020_NeuronClasses.xlsx", "LowResAtlasWithHighResHeadsAndTails.csv", "Hobert2016_BrainAtlas.xlsx"
]

# Datasets created with the (no longer existing, but in worm-graph)
# `CreateSyntheticDataset.ipynb` notebook.
SYNTHETIC_DATASETS = {  
    "Sines0000",
    "Lorenz0000",
    "WhiteNoise0000",
    "RandWalk0000",
    "VanDerPol0000",
    "Wikitext0000",
}

# Select correct device for torch
def init_device():
    """
    Initialize the global computing device to be used.
    """
    if torch.backends.mps.is_available():
        print("MPS device found.")
        device = torch.device("mps")
    elif torch.cuda.is_available():
        print("CUDA device found.")
        device = torch.device("cuda")
        gpu_props = torch.cuda.get_device_properties(device)
        print(f"\t GPU: {gpu_props.name}")
    else:
        print("Defaulting to CPU.")
        device = torch.device("cpu")
    return device

# Method for globally setting all random seeds
def init_random_seeds(seed=0):
    """
    Initialize random seeds for random, numpy, torch and cuda.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    return None


# won't be called at import time
if __name__ == "__main__":
    # can import init_device() or DEVICE if needed for processing in future
    DEVICE = init_device()