from preprocess._utils import *

import time


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
        multithread_flag = "with multithreading" if config['use_multithreading'] else "sequentially"
        logger.info(f"Preprocessing C. elegans neural data {multithread_flag} ({len(EXPERIMENT_DATASETS)} datasets)...")
        kwargs = dict(
            alpha=config["smooth"]["alpha"],
            window_size=config["smooth"]["window_size"],
            sigma=config["smooth"]["sigma"],
        )
        start_time = time.time()
        pickle_neural_data(
            url=config["opensource_neural_url"],
            zipfile=config["opensource_neural_zipfile"],
            source_dataset=config["source_dataset"],
            smooth_method=config["smooth"]["method"],
            resample_dt=config["resample_dt"],
            interpolate_method=config["interpolate"],
            cleanup=config["cleanup"],
            use_multithreading=config['use_multithreading'],
            **kwargs,
        )
        end_time = time.time()

        print("") # new line
        logger.info(f"Finished preprocessing neural data in {end_time - start_time:.2f} seconds.")
    else:
        logger.info("Neural data already preprocessed.")

    return None
