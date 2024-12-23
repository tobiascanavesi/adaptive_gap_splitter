import numpy as np
from adaptive_gap_splitter import percentile_gap_split_1d, percentile_gap_split_multidim

def example_1d():
    print("1D Example:")
    x = np.array([1, 2, 3, 4, 5, 10, 11, 13, 15, 42, 45, 49, 50])
    groups = percentile_gap_split_1d(x, percentile=95, min_group_size=2)
    for idx, group in enumerate(groups, 1):
        print(f"Group {idx}: {group}")

def example_multidim():
    print("\nMulti-Dimensional Example:")
    data_2d = np.array([
        [1, 2],
        [2, 2],
        [3, 4],
        [4, 5],
        [10, 10],
        [11, 11],
        [13, 13],
        [15, 15],
        [42, 40],
        [45, 46],
        [49, 48],
        [50, 49]
    ])
    clusters = percentile_gap_split_multidim(data_2d, percentile=95, min_group_size=2)
    for idx, cluster in enumerate(clusters, 1):
        print(f"Group {idx}:\n{cluster}\n")

if __name__ == "__main__":
    example_1d()
    example_multidim()
