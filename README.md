# Adaptive Gap Splitter

**Adaptive Gap Splitter** is a Python package designed to partition data into meaningful groups based on adaptive gap criteria. It dynamically determines split thresholds using percentile-based methods, making it flexible and robust without relying on fixed numeric thresholds or existing clustering algorithms.

## Features

- **1D Splitting**: Split one-dimensional data based on adaptive percentile gaps.
- **Multi-Dimensional Splitting**: Extend the splitting logic to multi-dimensional data by selecting dimensions with the highest variance.
- **Customizable Parameters**: Adjust percentiles and minimum group sizes to suit your data.
- **Recursive Splitting**: Automatically split data into hierarchical groups based on gap significance.

## Installation

You can install the package via `pip`:

```bash
pip install adaptive_gap_splitter

Note: Ensure that you have numpy installed, as it's a dependency.

Usage
1D Example

import numpy as np
from adaptive_gap_splitter import percentile_gap_split_1d

# Sample data
x = np.array([1, 2, 3, 4, 5, 10, 11, 13, 15, 42, 45, 49, 50])

# Apply the percentile-based adaptive gap splitting
groups = percentile_gap_split_1d(x, percentile=95, min_group_size=2)

# Display the results
for idx, group in enumerate(groups, 1):
    print(f"Group {idx}: {group}")

Output:

Group 1: [ 1  2  3  4  5 10 11 13 15]
Group 2: [42 45 49 50]

Multi-Dimensional Example

import numpy as np
from adaptive_gap_splitter import percentile_gap_split_multidim

# Sample 2D data
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

# Apply the percentile-based adaptive gap splitting
clusters = percentile_gap_split_multidim(data_2d, percentile=95, min_group_size=2)

# Display the results
for idx, cluster in enumerate(clusters, 1):
    print(f"Group {idx}:\n{cluster}\n")

Output:

Group 1:
[[ 1  2]
 [ 2  2]
 [ 3  4]
 [ 4  5]
 [10 10]
 [11 11]
 [13 13]
 [15 15]]

Group 2:
[[42 40]
 [45 46]
 [49 48]
 [50 49]]

API Reference
percentile_gap_split_1d(data, percentile=95, min_group_size=2)

Recursively splits 1D data based on adaptive percentile gaps.

    Parameters:
        data (array-like): 1D numerical data.
        percentile (float): Percentile to set the gap threshold (0-100).
        min_group_size (int): Minimum number of points in a group to consider splitting.

    Returns:
        List[np.ndarray]: A list of numpy arrays, each representing a group.

percentile_gap_split_multidim(data, percentile=95, min_group_size=2)

Recursively splits multi-dimensional data based on adaptive percentile gaps in the dimension with the highest variance.

    Parameters:
        data (2D array-like): Multi-dimensional numerical data with shape (n_samples, n_features).
        percentile (float): Percentile to set the gap threshold (0-100).
        min_group_size (int): Minimum number of points in a group to consider splitting.

    Returns:
        List[np.ndarray]: A list of numpy arrays, each representing a group.

License

This project is licensed under the MIT License. See the LICENSE file for details.
Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
Contact

For any questions or suggestions, please open an issue on the GitHub repository.