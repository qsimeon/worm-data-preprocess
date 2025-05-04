# Supplementary Scripts

This folder contains all additional scripts related to the C. elegans neural analysis project:

## Script Descriptions

### `consensus_connectome.py`
Generates CSV files for consensus connectome data available on the [HuggingFace repository](https://huggingface.co/datasets/qsimeon/celegans_connectome_data).

### `neural_to_parquet.py`
Generates the parquet file for neural activity data available on the [HuggingFace repository](https://huggingface.co/datasets/qsimeon/celegans_neural_data). 

---

### `visuals/visualize_connectome.py`
Contains additional code for connectome data visualization and verification; helpful for iteration.

### `visuals/plot_amphid_consensus.py`
Visualizes the consensus connectome for the 22 amphid chemosensory neurons using
the updated CSV output. Produces a directed graph image showing aggregated edge
weights. Used as Appendix Fig. 1 in preprint.