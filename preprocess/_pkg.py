import os
import json
import h5py
import math
import torch
import mat73
import shutil
import pickle
import zipfile
import logging
import subprocess
import multiprocessing
from urllib.request import urlopen
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from collections import defaultdict

# NOTE: IterativeImputer is experimental and the API might change without any deprecation cycle.
# To use it, you need to explicitly import enable_iterative_imputer.
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# NOTE: interp1d is considered legacy and may be removed in future updates of scipy.
from scipy.interpolate import interp1d

from pynwb import NWBHDF5IO
from scipy.io import loadmat
from typing import Dict, List, Tuple, Set
from sklearn import preprocessing
from scipy.ndimage import gaussian_filter1d
from torch_geometric.data import Data, download_url as tg_download_url
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch_geometric.utils import coalesce, to_dense_adj, dense_to_sparse

# Local variables
from _main_utils import (
    RAW_ZIP,
    ROOT_DIR,
    RAW_FILES,
    NUM_NEURONS,
    NEURON_LABELS,
    RAW_DATA_URL,
    RAW_DATA_DIR,
)
from preprocess.config import PREPROCESS_CONFIG
EXPERIMENT_DATASETS = PREPROCESS_CONFIG['EXPERIMENT_DATASETS'] # explicit so imports continue to work

# # Configure logger
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name) - %(levelname)s - %(message)s',
#     datefmt='%H:%M:%S'
# )
# # Create a logger for each module
logger = logging.getLogger(__name__)