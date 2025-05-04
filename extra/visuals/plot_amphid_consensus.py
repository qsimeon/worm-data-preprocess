
"""
Generates a directed graph visualization of the 22 amphid chemosensory neurons
using the updated consensus_connectome_full_nofunc.csv file.

Place script in root directory for it to properly find dataset.
"""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os

# Define the set of amphid chemosensory neurons (22 total, 11 bilateral pairs)
amphid_neurons = {
    "ADF", "ADL", "ASE", "ASG", "ASH", "ASI", "ASJ", "ASK", "AWA", "AWB", "AWC"
}
sides = {"L", "R"}
amphid_full_set = {f"{n}{s}" for n in amphid_neurons for s in sides}

# Load the consensus connectome CSV
# Define the root directory relative to this script's location
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_FILE_NAME = "consensus_connectome_full_nofunc.csv"
# Construct the path to the graph file
DATA_FILE_PATH = os.path.join(
    ROOT_DIR, "datasets", DATA_FILE_NAME
)
if not os.path.exists(DATA_FILE_PATH):
    print(f"ERROR: Graph file not found at: {DATA_FILE_PATH}")
    # Stop execution if file not found
    raise FileNotFoundError(f"Graph file not found at: {DATA_FILE_PATH}")

df = pd.read_csv(DATA_FILE_PATH)

# Filter rows where both from_neuron and to_neuron are amphid chemosensory neurons
df_subset = df[
    df["from_neuron"].isin(amphid_full_set) & df["to_neuron"].isin(amphid_full_set)
]

# Create a directed graph
G = nx.DiGraph()

# Add edges with weights (gap and chemical averaged if both exist)
for _, row in df_subset.iterrows():
    weight = 0
    if not pd.isna(row["mean_gap_weight"]):
        weight += row["mean_gap_weight"]
    if not pd.isna(row["mean_chem_weight"]):
        weight += row["mean_chem_weight"]
    if weight > 0:
        G.add_edge(row["from_neuron"], row["to_neuron"], weight=weight)

# Positioning with spring layout
pos = nx.spring_layout(G, seed=42)

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_color="lightyellow", edgecolors="black", node_size=800)

# Draw labels
nx.draw_networkx_labels(G, pos, font_size=8)

# Draw edges with widths proportional to weight
edges = G.edges(data=True)
weights = [d["weight"] for (_, _, d) in edges]
nx.draw_networkx_edges(G, pos, edge_color="red", width=[w / max(weights) * 3 for w in weights], arrows=True)

# Finalize and show plot
# plt.title("Consensus Connectome of Amphid Chemosensory Neurons")
plt.axis("off")
plt.tight_layout()
plt.savefig("amphid_consensus_graph.png", dpi=300)
plt.show()
