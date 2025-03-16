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
   _The size of the environment is ~900MB._

---

## Step 2: Preprocessing

1. To prepare for a preprocessing run, update your configuration in `preprocess/config.py`.

2. To download and preprocess the neural data, run:

```bash
python preprocess.py (--verbose)
```

_Downloading the neural data should take approximately 10 minutes, depending on your download speed._

_Preprocessing all the data takes approximately 11 mins to run on an M2 MacBook Pro, but will be highly dependent on your on your hardware. We recommend enabling multithreading in the `config.py` file to make this as fast as possible._

3. Once the data has been preprocessed, you will find the processed data in
   `data/processed/`

4. Once you have selected the files you need, you may clear all processed and
   downloaded files using:

```bash
python cleanup.py (--force)
```

_This clears around 40gb of files._

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

## Datasets

The following 12 datasets are included by default:

| Index | Dataset Name        | Num. Worms | Mean Num. Neurons (labeled, recorded) |
| ----- | ------------------- | ---------- | ------------------------------------- |
| 1     | `Kato2015  `        | 12         | (42, 128)                             |
| 2     | `Nichols2017`       | 44         | (35, 108)                             |
| 3     | `Skora2018`         | 12         | (47, 129)                             |
| 4     | `Kaplan2020  `      | 19         | (37, 115)                             |
| 5     | `Nejatbakhsh2020 `  | 21         | (174, 175)                            |
| 6     | `Yemini2021 `       | 49         | (111, 111)                            |
| 7     | `Uzel2022    `      | 6          | (50, 139)                             |
| 8     | `Dag2023`           | 7          | (101, 143)                            |
| 9     | `Leifer2023`        | 103        | (64, 68)                              |
| 10    | `Lin2023`           | 577        | (8, 8)                                |
| 11    | `Flavell2023`       | 40         | (89, 136)                             |
| 12    | `Venkatachalam2024` | 22         | (187, 187)                            |

See how this table was generated from the final parquet
[here](https://colab.research.google.com/drive/1z7h2gGuWhupRtjpYc7IHFD4rJ4kIsyuD#scrollTo=ZiZXMRc931oy)

---

## Preliminary Data Analysis

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
