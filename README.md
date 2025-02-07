# worm-data-preprocess
Repository to manage the data preprocessing of C. elegans neural activity and
connectome data to be used for downstream analysis and machine learning tasks.


# Installation Guide
## Prerequisites
- Ensure you have **Conda** installed. If not, download and install it from [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
- Install a Python version compatible with the listed libraries (e.g., Python 3.8+).

---

## Step 1: Create a Conda Environment

1. Open a terminal or command prompt.
2. Create a new Conda environment using the provided `environment.yml` file:
   ```bash
   conda env create -f environment.yml
   ```

3. Activate the environment:
   ```bash
   conda activate worm-preprocess
   ```
*The size of the environment is ~900MB.*


---

## Step 2: Preprocessing

1. To prepare for a preprocessing run, update your configuration in `preprocess/config.py`.

2. To download and preprocess the neural data, run:
```bash
python preprocess.py
```
*Downloading the neural data can take up to 10 minutes, depending on your download speed.*

*Preprocessing all the data should take approximately 3-5 mins to run, depending*
*on your hardware. We recommend enabling multithreading in the `config.py` file to speed this up.*

3. Once the data has been preprocessed, you will find the processed data in
   `data/processed/`

4. Once you have selected the files you need, you may clear all processed and
   downloaded files using:
```bash
python cleanup.py
``` 

*This clears around 40gb of files.*

---

## (optional) Step 3: Generate Parqet File

Optionally, we have included the ability to generate a parquet file for
clear of viewing the data. This is what is used to create the data file
available on the associated HuggingFace repo.

1. Run the script:
```bash
python neural_to_parqet.py
``` 
2. Access the generated parqet file at `data/datasets/preprocessed_worm_data_short.parqet`

---
