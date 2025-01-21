from ._utils import *


def process_data(config: dict) -> None:
    """Preprocesses the raw neural and connectome data.

    This function preprocesses raw neural and connectome data to be used
    in downstream modeling and analysis tasks. It checks if the neural
    data and connectome data have been processed already. If not, it calls
    the appropriate functions to process and save them in the specified format.

    Params
    ------
    config: dict
        Configuration dictionary. See config.py for details.

    Calls
    -----
    pickle_neural_data : function in preprocess/_utils.py
    preprocess_connectome : function in preprocess/_utils.py
    """
    # Initialize the logger
    logger = logging.getLogger(__name__)

    # Download and pickle the neural data if not already done
    if not os.path.exists(os.path.join(ROOT_DIR, "data/processed/neural/.processed")):
        logger.info("Preprocessing C. elegans neural data...")
        kwargs = dict(
            alpha=config["smooth"]["alpha"],
            window_size=config["smooth"]["window_size"],
            sigma=config["smooth"]["sigma"],
        )
        pickle_neural_data(
            url=config["opensource_url"],
            zipfile=config["opensource_zipfile"],
            source_dataset=config["source_dataset"],
            smooth_method=config["smooth"]["method"],
            resample_dt=config["resample_dt"],
            interpolate_method=config["interpolate"],
            cleanup=config["cleanup"],
            **kwargs,
        )
        logger.info("Finished preprocessing neural data.")
    else:
        logger.info("Neural data already preprocessed.")

    # Extract presaved commonly used neural dataset split patterns
    if not os.path.exists(os.path.join(ROOT_DIR, "data/datasets")):
        logger.info("Extracting presaved dataset patterns.")
        get_presaved_datasets(
            url=config["presaved_url"], file=config["presaved_file"]
        )
        logger.info("Done extracting presaved dataset patterns.")
    else:
        logger.info("Presaved dataset patterns already extracted.")

    # Preprocess the connectome data if not already done
    if not os.path.exists(os.path.join(ROOT_DIR, "data/processed/connectome/.processed")):
        logger.info("Preprocessing C. elegans connectome data ...")
        preprocess_connectome(
            raw_files=RAW_FILES, source_connectome=config["connectome_pub"]
        )
        logger.info("Finished preprocessing connectome.")
    else:
        logger.info("Connectome already preprocessed.")
    return None


if __name__ == "__main__":
    print("Configuration:", json.dumps(PREPROCESS_CONFIG, indent=2), end="\n\n")
    process_data(PREPROCESS_CONFIG)
