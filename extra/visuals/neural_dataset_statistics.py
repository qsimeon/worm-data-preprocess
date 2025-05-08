# Dataset Info Visualization Notebook
# This notebook reproduces and updates Figure 1 from the preprint, including the new dataset Nejatbakhsh2020.

# Figure 1:
# (A) Distribution of the number of worms in each experimental dataset. (B) Number of recorded neurons per worm compared to the total neuronal population of C. elegans. (C) Total duration of recorded neural activity for each dataset. (D) Average recording duration per worm, with one hour of calcium imaging as a benchmark. (E) Number of time steps per worm, and (F) Pre-resampled sampling intervals for recorded neural activity. The horizontal dashed line in (F) indicates the target resampled time step (∆t = 0.333 seconds) used in our preprocessing pipeline.   

## 1. Imports and settings
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


## 2. Load data
# Absolute path based on script location
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_FILE_NAME = "processed_worm_data_latest.parquet"
parquet_path = os.path.join(ROOT_DIR, "datasets", DATA_FILE_NAME)
print(f"Loading data from: {parquet_path}")

# Read only the desired columns
df = pd.read_parquet(
    parquet_path,
    columns=[
        "source_dataset",
        "worm",
        "max_timesteps",  # sampled time steps per worm
        "original_time_in_seconds",  # original per-worm time vector
        "time_in_seconds",  # processed per-worm time vector
    ],
)

## 3. Data preprocessing
# A: Number of worms per dataset
n_worms = df.groupby("source_dataset")["worm"].nunique()

# B: Neurons recorded per worm (mean)
per_worm_neurons = df.groupby(["source_dataset", "worm"]).size()
ne_per_worm = per_worm_neurons.groupby("source_dataset").mean()

# C & D: Durations per worm (seconds)
# Extract one row per worm to get unique time vectors
per_worm_df = df.drop_duplicates(subset=['source_dataset', 'worm']).copy()

# Compute duration per worm as the max of original_time_in_seconds array
per_worm_df['duration_s'] = per_worm_df['original_time_in_seconds'].apply(lambda arr: np.max(arr))

# C: Total duration per dataset
total_duration_s = per_worm_df.groupby('source_dataset')['duration_s'].sum()

# D: Average duration per worm
avg_duration_s = per_worm_df.groupby('source_dataset')['duration_s'].mean()

# E: Sampled time steps per worm (mean)
avg_timesteps = per_worm_df.groupby('source_dataset')['max_timesteps'].mean()


# F: Sampling interval (median ∆t per worm, then averaged)
def compute_dt_median(time_vec):
    diffs = np.diff(time_vec)
    return np.median(diffs) if diffs.size > 0 else np.nan


# F: Sampling interval (median ∆t per worm, then averaged)
dt_records = []
for _, row in per_worm_df.iterrows():
    ds = row['source_dataset']
    time_vec = row['time_in_seconds']
    dt_records.append({
        'source_dataset': ds,
        'dt_median': compute_dt_median(time_vec)
    })
dt_df = pd.DataFrame(dt_records)
avg_dt = dt_df.groupby('source_dataset')['dt_median'].mean()

# Benchmarks for panels E and F
dt_bench = 0.333  # resampled time step (s)

## 4. Plotting
# Create figure with a specific layout to accommodate the legend
fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(2, 4, wspace=0.4, hspace=0.3)  # Add spacing between subplots

# Create subplots
axes = [
    fig.add_subplot(gs[0, 0]),  # Panel A (pie)
    fig.add_subplot(gs[0, 1]),  # Panel B
    fig.add_subplot(gs[0, 2]),  # Panel D (moved up)
    fig.add_subplot(gs[1, 0]),  # Panel C (moved down)
    fig.add_subplot(gs[1, 1]),  # Panel E
    fig.add_subplot(gs[1, 2]),  # Panel F
]

# Prepare color palette
datasets = n_worms.index.tolist()
colors = plt.get_cmap("tab20").colors[: len(datasets)]

# Panel A: Worm counts
axes[0].pie(n_worms[datasets], labels=None, colors=colors, 
            autopct=lambda p: f'{int(p*n_worms.sum()/100)}' if p > 3 else '',  # Show count if slice > 3%
            pctdistance=1.2,  # Move percentages outside
            textprops={'fontsize': 8})  # Smaller font size
axes[0].set_title("(A) Number of worms per dataset")

# Panel B: Neurons per worm
sorted_indices = np.argsort(ne_per_worm[datasets])[::-1]  # Descending order
axes[1].bar(range(len(datasets)), ne_per_worm[datasets].iloc[sorted_indices], 
            color=[colors[i] for i in sorted_indices])
axes[1].axhline(302, ls="--", color='red', label="Total C. elegans neurons")
axes[1].set_ylabel("Neurons per worm")
axes[1].set_title("(B) Recorded neurons per worm")
axes[1].set_xticks([])  # Remove x-axis ticks
axes[1].legend()

# Panel D: Average duration per worm (moved up)
sorted_indices = np.argsort(avg_duration_s[datasets])[::-1]  # Descending order
axes[2].bar(range(len(datasets)), avg_duration_s[datasets].iloc[sorted_indices], 
            color=[colors[i] for i in sorted_indices])
axes[2].axhline(3600, ls="--", label="1 hour benchmark")
axes[2].set_ylabel("Average recording duration (s)")
axes[2].set_title("(D) Average recording duration per worm")
axes[2].set_xticks([])  # Remove x-axis ticks
axes[2].legend()

# Panel C: Total duration (seconds) (moved down)
axes[3].pie(
    total_duration_s[datasets], labels=None, colors=colors,
    autopct=lambda p: f'{p:.1f}%' if p > 5 else '',  # Only show percentages > 5%
    pctdistance=1.2,  # Move percentages outside
    textprops={'fontsize': 8})  # Smaller font size
axes[3].set_title("(C) Total duration (seconds)")

# Panel E: Sampled time steps per worm
sorted_indices = np.argsort(avg_timesteps[datasets])[::-1]  # Descending order
axes[4].bar(range(len(datasets)), avg_timesteps[datasets].iloc[sorted_indices], 
            color=[colors[i] for i in sorted_indices])
axes[4].set_ylabel("Resampled time steps per worm")
axes[4].set_title("(E) Resampled time steps per worm")
axes[4].set_xticks([])  # Remove x-axis ticks

# Panel F: Sampling interval
sorted_indices = np.argsort(avg_dt[datasets])[::-1]  # Descending order
axes[5].bar(range(len(datasets)), avg_dt[datasets].iloc[sorted_indices], 
            color=[colors[i] for i in sorted_indices])
axes[5].axhline(dt_bench, ls="--", label=f"Δt = {dt_bench} s")
axes[5].set_ylabel("Pre-Resampled Sampling interval (s)")
axes[5].set_title("(F) Pre-Resampled Sampling interval")
axes[5].set_xticks([])  # Remove x-axis ticks
axes[5].legend()

# Create a dedicated subplot for the legend
legend_ax = fig.add_subplot(gs[:, 3])  # Span both rows in the last column
legend_ax.axis('off')  # Hide the axes
handles = [plt.Rectangle((0,0),1,1, color=color) for color in colors]
legend_ax.legend(handles, datasets, loc='center left', title="Datasets")

# save this to /visuals
plt.savefig("extra/visuals/dataset_figure1_updated.png", dpi=300, bbox_inches='tight')
plt.show()
