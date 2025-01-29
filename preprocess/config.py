PREPROCESS_CONFIG = {
    ## NEURAL DATA
    # URL to download open source worm behavior dataset
    "opensource_neural_url": "https://www.dropbox.com/scl/fi/vfygz1twi1jg62cfssc0w/opensource_data.zip?rlkey=qa4vpwcoza3k9v5o2watwblth&dl=1",
    # Local filename for downloaded opensource neural data
    "opensource_neural_zipfile": "opensource_neural_data.zip",
    
    ## PREPROCESSED DATA (for training models)
    # URL to download pre-processed datasets
    "presaved_url": "https://www.dropbox.com/scl/fi/baikxamldjyrf5maephk3/presaved_datasets.zip?rlkey=4qrso6forjpvfdbm9mll3ndxf&dl=1",
    # Local filename for pre-processed data
    "presaved_file": "presaved_datasets.zip",
    
    ## CONNECTOME DATA
    # Connectome publication to use ('all' or specific publication name)
    "connectome_pub": "all",
    
    # Whether to delete downloaded data after processing
    "cleanup": False,
    # Elect to use multithreading (default True)
    "use_multithreading": False,
    # Dataset selection ('all' or specific dataset names)
    "source_dataset": "all",
    # Time interval in seconds for resampling neural activity
    "resample_dt": 0.333,
    # Interpolation method for missing data points
    "interpolate": "linear",
    # Signal smoothing parameters
    "smooth": {
        # Smoothing method (none/gaussian/exponential/moving)
        "method": "none",
        # Exponential smoothing factor (smaller = smoother)
        "alpha": 0.5,
        # Gaussian kernel width (larger = smoother)
        "sigma": 5,
        # Moving average window size (larger = smoother)
        "window_size": 15
    }
}
