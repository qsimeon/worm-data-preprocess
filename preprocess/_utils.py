from preprocess._pkg import *
from preprocess.preprocessors._base_preprocessors import *
from preprocess.preprocessors._neural import *
from preprocess.preprocessors._connectome import *

# Initialize logger
logger = logging.getLogger(__name__)

def download_url_with_progress(url, folder, log=True, filename=None):
    """
    A wrapper for torch_geometric's download_url to add a simple binary
    progress bar and optionally log the file size before starting.
    """
    # Try to fetch the file size
    try:
        info = urlopen(url).info() # urllibobject with .info() dict
        file_size_mb = f"{int(info['Content-Length']) / (1024 * 1024):.2f} MB" if 'Content-Length' in info else "unknown"
    except Exception as e:
        file_size_mb = "unknown"

    # Proceed with the original download logic
    if log is True:
        with tqdm(
            desc=f"Downloading {filename if filename else url} (size: {file_size_mb})", 
            total=1, 
            ncols=100, 
            unit="file"
        ) as progress:
            result = tg_download_url(url=url, folder=folder, log=False, filename=filename)
            progress.update(1)
    else:
        result = tg_download_url(url=url, folder=folder, log=False, filename=filename)

    return result


def process_single_dataset(args):
    """Helper function to process a single dataset

    Args:
        args (tuple): (source, transform, smooth_method, interpolate_method, resample_dt, kwargs)
    """
    source, transform, smooth_method, interpolate_method, resample_dt, kwargs = args
    try:
        logger.info(f"Start processing {source}.")
        # Instantiate the relevant preprocessor class (dynamic class evaluation)
        preprocessor = eval(source + "Preprocessor")(
            transform,
            smooth_method,
            interpolate_method,
            resample_dt,
            **kwargs,
        )
        # Call its method
        preprocessor.preprocess()
        return True
    except NameError as e:
        logger.info(f"NameError calling preprocessor: {e}")
        return False


def pickle_neural_data(
    url,
    zipfile,
    source_dataset="all",
    transform=StandardScaler(),
    smooth_method="none",
    interpolate_method="linear",
    resample_dt=None,
    cleanup=False,
    n_workers=None,  # New parameter for controlling number of workers
    **kwargs,
):
    """Preprocess and save C. elegans neural data to .pickle format.

    This function downloads and extracts the open-source datasets if not found in the
    root directory, preprocesses the neural data using the corresponding DatasetPreprocessor class,
    and then saves it to .pickle format. The processed data is saved in the
    data/processed/neural folder for further use.

    Args:
        url (str): Download link to a zip file containing the open-source data in raw form.
        zipfile (str): The name of the zipfile that is being downloaded.
        source_dataset (str, optional): The name of the source dataset to be pickled.
            If None or 'all', all datasets are pickled. Default is 'all'.
        transform (object, optional): The sklearn transformation to be applied to the data.
            Default is StandardScaler().
        smooth_method (str, optional): The smoothing method to apply to the data;
            options are 'gaussian', 'exponential', or 'moving'. Default is 'moving'.
        interpolate_method (str, optional): The scipy interpolation method to use when resampling the data.
            Default is 'linear'.
        resample_dt (float, optional): The resampling time interval in seconds.
            If None, no resampling is performed. Default is None.
        cleanup (bool, optional): If True, deletes the unzipped folder after processing. Default is False.
        **kwargs: Additional keyword arguments to be passed to the DatasetPreprocessor class.

    Returns:
        None

    Raises:
        AssertionError: If an invalid source dataset is requested.
        NameError: If the specified preprocessor class is not found.

    Steps:
        1. Construct paths for the zip file and source data.
        2. Create the neural data directory if it doesn't exist.
        3. Download and extract the zip file if the source data is not found.
        4. Instantiate and use the appropriate DatasetPreprocessor class to preprocess the data.
        5. Save the preprocessed data to .pickle format.
        6. Optionally, delete the unzipped folder if cleanup is True.
    """
    zip_path = os.path.join(ROOT_DIR, zipfile)
    source_path = os.path.join(ROOT_DIR, zipfile.strip(".zip"))
    # Make the neural data directory if it doesn't exist
    processed_path = os.path.join(ROOT_DIR, "data/processed/neural")
    if not os.path.exists(processed_path):
        os.makedirs(processed_path, exist_ok=True)
    # If .zip not found in the root directory, download the curated open-source worm datasets
    if not os.path.exists(source_path):
        try:
            # downloads opensource_neural_data
            download_url_with_progress(url=url, folder=ROOT_DIR, filename=zipfile)
        except Exception as e:
            logger.error(f"Failed to download using async method: {e}")
            logger.info("Falling back to wget...")
            # Fallback to wget if async download fails
            import subprocess

            subprocess.run(
                [
                    "wget",
                    "-O",
                    os.path.join(ROOT_DIR, zipfile),
                    "--tries=3",  # Retry 3 times
                    "--continue",  # Resume partial downloads
                    "--progress=bar:force",  # Show progress bar
                    url,
                ]
            )
        # Extract all the datasets ... OR
        if source_dataset.lower() == "all":
            # Extract zip file then delete it
            extract_zip(zip_path, folder=source_path, delete_zip=True)
        # Extract just the requested source dataset
        else:
            bash_command = [
                "unzip",
                zip_path,
                "{}/*".format(source_dataset),
                "-d",
                source_path,
                "-x",
                "__MACOSX/*",
            ]
            # Run the bash command
            std_out = subprocess.run(bash_command, text=True)
            # Output to log or terminal
            logger.info(f"Unzip status {std_out} ...")
            # Delete the zip file
            os.unlink(zip_path)
    # (re)-Pickle all the datasets ... OR
    if source_dataset is None or source_dataset.lower() == "all":
        # Determine number of workers (use CPU count - 1 by default)
        if n_workers is None:
            n_workers = max(1, multiprocessing.cpu_count() - 1)

        # Prepare arguments for parallel processing
        process_args = [
            (source, transform, smooth_method, interpolate_method, resample_dt, kwargs)
            for source in EXPERIMENT_DATASETS
        ]

        # Use multiprocessing Pool to process datasets in parallel
        with Pool(processes=n_workers) as pool:
            results = pool.map(process_single_dataset, process_args)

        # Create .processed file to indicate that some preprocessing was successful
        if any(results):  # If at least one dataset was processed successfully
            open(os.path.join(processed_path, ".processed"), "a").close()

    # ... (re)-Pickle a single dataset
    else:
        assert (
            source_dataset in EXPERIMENT_DATASETS
        ), "Invalid source dataset requested! Please pick one from:\n{}".format(
            list(EXPERIMENT_DATASETS)
        )
        process_single_dataset(
            (source_dataset, transform, smooth_method, interpolate_method, resample_dt, kwargs)
        )

    # Delete the unzipped folder
    if cleanup:
        shutil.rmtree(source_path)
    return None


def get_presaved_datasets(url, file):
    """Download and unzip presaved data patterns.

    This function downloads and extracts presaved data patterns).
    from the specified URL. The extracted data is saved in the 'data' folder.
    The zip file is deleted after extraction.

    Args:
        url (str): The download link to the zip file containing the presaved data splits.
        file (str): The name of the zip file to be downloaded.

    Returns:
        None

    Steps:
        1. Construct the paths for the zip file and the data directory.
        2. Download the zip file from the specified URL.
        3. Extract the contents of the zip file to the data directory.
        4. Delete the zip file after extraction.
    """
    presaved_url = url
    presaved_file = file
    presave_path = os.path.join(ROOT_DIR, presaved_file)
    data_path = os.path.join(ROOT_DIR, "data")
    download_url_with_progress(url=presaved_url, folder=ROOT_DIR, filename=presaved_file)
    extract_zip(presave_path, folder=data_path, delete_zip=True)
    return None


def preprocess_connectome(raw_files, source_connectome=None):
    """Convert the raw connectome data to a graph tensor.

    This function processes raw connectome data, which includes chemical
    synapses and gap junctions, into a format suitable for use in machine
    learning or graph analysis. It reads the raw data in tabular format (.csv, .xls[x]),
    processes it to extract the relevant information, and creates graph
    tensors that represent the C. elegans connectome. The resulting
    graph tensors are saved in the 'data/processed/connectome' folder
    as 'graph_tensors.pt'. We distinguish between electrical (gap junction)
    and chemical synapses by using an edge attribute tensor with two feature dimensions:
    the first feature represents the weight of the gap junctions; and the second feature
    represents the weight of the chemical synapses.

    Args:
        raw_files (list): Contain the names of the raw connectome data to preprocess.
        source_connectome (str, optional): The source connectome file to use for preprocessing. Options include:
            - "openworm": OpenWorm project  (augmentation of earlier connectome with neurotransmitter type)
            - "funconn" or "randi_2023": Randi et al., 2023 (functional connectivity)
            - "witvliet_7": Witvliet et al., 2020 (adult 7)
            - "witvliet_8": Witvliet et al., 2020 (adult 8)
            - "white_1986_whole": White et al., 1986 (whole)
            - "white_1986_n2u": White et al., 1986 (N2U)
            - "white_1986_jsh": White et al., 1986 (JSH)
            - "white_1986_jse": White et al., 1986 (JSE)
            - "cook_2019": Cook et al., 2019
            - "all": preprocess all of the above connectomes separately
            - None: Default to a preprocessed variant of Cook et al., 2019

    Returns:
        None

    Steps:
        1. Check that all necessary files are present.
        2. Download and extract the raw data if not found.
        3. Determine the appropriate preprocessing class based on the publication.
        4. Instantiate and use the appropriate preprocessor class to preprocess the data.
        5. Save the preprocessed graph tensors to a file.

    NOTE:
    * A connectome is a comprehensive map of the neural connections within an
      organism's brain or nervous system. It is essentially the wiring diagram
      of the brain, detailing how neurons and their synapses are interconnected.
    """
    # Check that all necessary files are present
    all_files_present = all([os.path.exists(os.path.join(RAW_DATA_DIR, rf)) for rf in raw_files])
    if not all_files_present:
        download_url_with_progress(url=RAW_DATA_URL, folder=ROOT_DIR, filename=RAW_ZIP)
        extract_zip(
            path=os.path.join(ROOT_DIR, RAW_ZIP),
            folder=RAW_DATA_DIR,
            delete_zip=True,
        )

    # Make the connectome data directory if it doesn't exist
    processed_path = os.path.join(ROOT_DIR, "data/processed/connectome")
    if not os.path.exists(processed_path):
        os.makedirs(processed_path, exist_ok=True)

    # Determine appropriate preprocessing class based on publication
    preprocessors = {
        "openworm": OpenWormPreprocessor,
        "chklovskii": ChklovskiiPreprocessor,
        "funconn": Randi2023Preprocessor,
        "randi_2023": Randi2023Preprocessor,
        "witvliet_7": Witvliet2020Preprocessor7,
        "witvliet_8": Witvliet2020Preprocessor8,
        "white_1986_whole": White1986WholePreprocessor,
        "white_1986_n2u": White1986N2UPreprocessor,
        "white_1986_jsh": White1986JSHPreprocessor,
        "white_1986_jse": White1986JSEPreprocessor,
        "cook_2019": Cook2019Preprocessor,
        None: DefaultPreprocessor,
    }

    # Preprocess all the connectomes including the default one
    if source_connectome == "all":
        for preprocessor_class in preprocessors.values():
            preprocessor_class().preprocess()
        # Create a file to indicate that the preprocessing was successful
        open(os.path.join(processed_path, ".processed"), "a").close()

    # Preprocess just the requested connectome
    else:
        preprocessor_class = preprocessors.get(source_connectome, DefaultPreprocessor)
        preprocessor_class().preprocess()

    return None


def extract_zip(path: str, folder: str = None, log: bool = True, delete_zip: bool = True):
    """Extracts a zip archive to a specific folder while ignoring the __MACOSX directory.

    Args:
        path (str): The path to the zip archive.
        folder (str, optional): The folder where the files will be extracted to. Defaults to the parent of `path`.
        log (bool, optional): If False, will not log anything to the console. Default is True.
        delete_zip (bool, optional): If True, will delete the zip archive after extraction. Default is True.

    Steps:
        1. Determine the extraction folder. If not provided, use the parent directory of the zip file.
        2. Log the extraction process if logging is enabled.
        3. Open the zip file and iterate through its members.
            - Skip any members that are part of the __MACOSX directory.
            - Extract the remaining members to the specified folder.
        4. Delete the zip file if `delete_zip` is True.
    """
    if folder is None:
        folder = os.path.dirname(path)
    zip_filename = os.path.basename(path)
    if log:
        logger.info(f"Extracting {zip_filename}...")
    with zipfile.ZipFile(path, "r") as zip_ref:
        for member in zip_ref.namelist():
            if not member.startswith("__MACOSX/"):
                zip_ref.extract(member, folder)
    if delete_zip:
        os.unlink(path)



# # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
