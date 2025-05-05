PREPROCESS_CONFIG = {
    ## NEURAL DATA
    # URL to download open source worm behavior dataset
    "opensource_neural_url": "https://www.dropbox.com/scl/fi/vfygz1twi1jg62cfssc0w/opensource_data.zip?rlkey=qa4vpwcoza3k9v5o2watwblth&dl=1",
    # Local filename for downloaded opensource neural data
    "opensource_neural_zipfile": "opensource_neural_data.zip",
    # Set of real C. elegans datasets with custom processors
    # Select specific ranges of datasets to process with list indexing
    "EXPERIMENT_DATASETS": [
        "Kato2015",
        "Nichols2017",
        "Skora2018",
        "Kaplan2020",
        "Nejatbakhsh2020",
        "Yemini2021",
        "Uzel2022",
        "Dag2023",
        "Leifer2023",  # Different type of dataset: stimulus-response.
        "Lin2023",
        "Flavell2023",
        "Venkatachalam2024",  # This is unpublished data. Downloaded from chemosensory-data.worm.world/.
    ][:],
    # Dataset selection ('all' or specific dataset names)
    "source_dataset": "all",
    ## PREPROCESSED DATA (for training models)
    # URL to download pre-processed datasets
    "presaved_url": "https://www.dropbox.com/scl/fi/baikxamldjyrf5maephk3/presaved_datasets.zip?rlkey=4qrso6forjpvfdbm9mll3ndxf&dl=1",
    # Local filename for pre-processed data
    "presaved_file": "presaved_datasets.zip",
    ## CONNECTOME DATA
    # Connectome publication to use ('all' or specific publication name)
    #     [
    #     "all",
    #     "openworm",
    #     "chklovskii",
    #     "randi_2023", # "funconn" is a synonym
    #     "witvliet_7",
    #     "witvliet_8",
    #     "white_1986_whole",
    #     "white_1986_n2u",
    #     "white_1986_jsh",
    #     "white_1986_jse",
    #     "cook_2019"
    # ]
    "connectome_pub": "all",
    ## PREPROCESS PARAMETERS
    # Time interval in seconds for resampling neural activity
    "resample_dt": 0.333,
    # Interpolation method for missing data points
    "interpolate": "linear",
    # Signal smoothing parameters
    "smooth": {
        # Smoothing method (none/gaussian/exponential/moving). none is fastest
        "method": "moving",
        # Exponential smoothing factor (smaller = smoother)
        "alpha": 0.5,
        # Gaussian kernel width (larger = smoother)
        "sigma": 5,
        # Moving average window size (larger = smoother)
        "window_size": 15,
    },
    # Type of normalization transformation used (causal/standard)
    "norm_transform": "standard",
    ## PROCESSING SPEEDUPS
    # Whether to delete downloaded data after processing
    "cleanup": False,
    # Elect to use multithreading for parallel processing (recommended: True)
    "use_multithreading": True,
}
