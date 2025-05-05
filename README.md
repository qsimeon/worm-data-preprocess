# worm-data-preprocess

Repository to manage the data preprocessing of C. elegans neural activity and
connectome data to be used for downstream analysis and machine learning tasks.

# Installation Guide

## Prerequisites

- Ensure you have **Conda** installed. If not, download and install it from [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
- Install a Python version compatible with the listed libraries (recommended: Python 3.12+).

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
   _The size of the environment is ~900MB._

---

## Step 2: Preprocessing

1. To prepare for a preprocessing run, update your configuration in `preprocess/config.py`.

2. To download and preprocess the **neural data**, run:

   ```bash
   python preprocess.py -n (--verbose)
   ```

   _Downloading the neural data should take approximately 10 minutes, depending on your download speed._

   _Preprocessing all the neural data takes approximately 11 minutes to run on an M2 MacBook Pro, but will be highly dependent on your hardware. We recommend enabling multithreading in the `config.py` file to make this as fast as possible._

   The processed neural data will be found in `data/processed/neural/`.

3. To download and preprocess the **connectome data**, run:

   ```bash
   python preprocess.py -c (--verbose)
   ```

   _Downloading and preprocessing the connectome data is typically much faster than the neural data._

   The processed connectome data will be found in `data/processed/connectome/`.

4. Once you have selected the files you need, you may clear all processed and
   downloaded files using:

   ```bash
   python cleanup.py (--force)
   ```

   _This clears around 40GB of files._

---

## (optional) Step 3: Generate Aggregate Files

Optionally, you can generate aggregate files for easier data analysis and sharing.

### Neural Data: Generate Parquet File

This creates a Parquet file for clear viewing of the neural data, as used on the associated [HuggingFace](https://huggingface.co/datasets/qsimeon/celegans_neural_data).

1. Run the script:

   ```bash
   python extra/neural_to_parquet.py
   ```

2. Access the generated Parquet file at `data/datasets/preprocessed_worm_data_{NAME}.parquet`

### Connectome Data: Generate Master CSV

This creates an aggregate CSV file for the connectome data, as used on the associated [HuggingFace](https://huggingface.co/datasets/qsimeon/celegans_connectome_data).

1. Run the script:

   ```bash
   python extra/consensus_connectome.py
   ```

2. Access the generated CSV file at `data/datasets/consensus_connectome_[tags].csv`

---

## Neural Activity Datasets

The following 12 datasets are included by default:

| Index | Dataset Name        | Num. Worms | Mean Num. Neurons (labeled, recorded) |
| ----- | ------------------- | ---------- | ------------------------------------- |
| 1     | `Kato2015  `        | 12         | (42, 127)                             |
| 2     | `Nichols2017`       | 44         | (34, 108)                             |
| 3     | `Skora2018`         | 12         | (46, 129)                             |
| 4     | `Kaplan2020  `      | 19         | (36, 114)                             |
| 5     | `Nejatbakhsh2020 `  | 21         | (173, 175)                            |
| 6     | `Yemini2021 `       | 49         | (110, 110)                            |
| 7     | `Uzel2022    `      | 6          | (50, 138)                             |
| 8     | `Dag2023`           | 7          | (100, 143)                            |
| 9     | `Leifer2023`        | 103        | (63, 67)                              |
| 10    | `Lin2023`           | 577        | (8, 8)                                |
| 11    | `Flavell2023`       | 40         | (88, 136)                             |
| 12    | `Venkatachalam2024` | 22         | (187, 187)                            |

See how this table was generated from the final parquet
[here](https://colab.research.google.com/drive/1z7h2gGuWhupRtjpYc7IHFD4rJ4kIsyuD#scrollTo=ZiZXMRc931oy)

---

## Preliminary Data Analysis on Neural Activity

**Data Analysis Notebook**

<a target="_blank" href="https://colab.research.google.com/drive/1I-8zUmtZ6dnAxf4nn2qMOXkpYwb4m6Xh?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

--

**Neural Modeling / Network Inference Notebook**

<a target="_blank" href="https://colab.research.google.com/drive/1DX0fPj0-pJYek48Xdmjpz42OjJzsaF9v?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

---

_This code is part of an ongoing research project with the MIT Department of Brain and
Cognitive Sciences in the Boyden Lab._
