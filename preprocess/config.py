# See preprocess/README.md for detailed documentation of all configuration options

PREPROCESS_CONFIG = {
    # Neural Data Sources
    "opensource_neural_url": "https://www.dropbox.com/scl/fi/vfygz1twi1jg62cfssc0w/opensource_data.zip?rlkey=qa4vpwcoza3k9v5o2watwblth&dl=1",
    "opensource_neural_zipfile": "opensource_neural_data.zip",
    "EXPERIMENT_DATASETS": [
        "Kato2015",
        "Nichols2017",
        "Skora2018",
        "Kaplan2020",
        "Nejatbakhsh2020",
        "Yemini2021",
        "Uzel2022",
        "Dag2023",
        "Leifer2023",
        "Lin2023",
        "Flavell2023",
        "Venkatachalam2024",
    ][:],
    "source_dataset": "all",

    # Connectome Configuration
    "connectome_pub": "all",

    # Preprocessing Parameters
    "resample_dt": 0.333,
    "interpolate": "linear",
    "smooth": {
        "method": "moving",
        "alpha": 0.5,
        "sigma": 5,
        "window_size": 15,
    },
    "norm_transform": "standard",

    # Processing Options
    "cleanup": False,
    "use_multithreading": True,
}
