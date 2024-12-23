import numpy as np

def percentile_gap_split_1d(data, percentile=95, min_group_size=2):
    """
    Recursively splits 1D data by finding the largest gap and comparing it
    to a percentile-based adaptive threshold.

    :param data: 1D array-like of numeric data
    :param percentile: percentile value (0-100) to set the threshold
    :param min_group_size: minimum size of a group to consider splitting
    :return: list of numpy arrays representing the final groups
    """
    data = np.array(data)
    
    # Base case: if too few points, just return as one group
    if len(data) < min_group_size:
        return [data]
    
    # 1) Sort the data
    data_sorted = np.sort(data)
    
    # 2) Compute consecutive gaps
    gaps = np.diff(data_sorted)
    
    # If no gaps (only one data point), return as one group
    if len(gaps) == 0:
        return [data_sorted]
    
    # 3) Determine the threshold as the specified percentile of gaps
    threshold = np.percentile(gaps, percentile)
    
    # 4) Find the largest gap
    max_gap_idx = np.argmax(gaps)
    max_gap_value = gaps[max_gap_idx]
    
    # 5) Compare the largest gap to the threshold
    if max_gap_value >= threshold and not np.all(gaps == max_gap_value):
        # Split the data at the largest gap
        left_group = data_sorted[:max_gap_idx + 1]
        right_group = data_sorted[max_gap_idx + 1:]
        
        # Recursively split each subgroup
        left_subgroups = percentile_gap_split_1d(left_group, percentile, min_group_size)
        right_subgroups = percentile_gap_split_1d(right_group, percentile, min_group_size)
        
        return left_subgroups + right_subgroups
    else:
        # No significant gap found; treat entire data as one group
        return [data_sorted]

def percentile_gap_split_multidim(data, percentile=95, min_group_size=2):
    """
    Recursively splits multi-dimensional data by finding the largest gap
    in the dimension with the highest variance and comparing it to a
    percentile-based adaptive threshold.

    :param data: 2D numpy array of shape (n_samples, n_features)
    :param percentile: percentile value (0-100) to set the threshold
    :param min_group_size: minimum size of a group to consider splitting
    :return: list of 2D numpy arrays representing the final groups
    """
    data = np.array(data)
    n_samples, n_features = data.shape
    
    # Base case: if too few points, just return as one group
    if n_samples < min_group_size:
        return [data]
    
    # 1) Select the dimension with the highest variance
    variances = np.var(data, axis=0)
    dim_to_split = np.argmax(variances)
    
    # 2) Sort the data based on the selected dimension
    sorted_indices = np.argsort(data[:, dim_to_split])
    data_sorted = data[sorted_indices]
    
    # 3) Compute consecutive gaps in the selected dimension
    gaps = np.diff(data_sorted[:, dim_to_split])
    
    # If no gaps (only one data point), return as one group
    if len(gaps) == 0:
        return [data_sorted]
    
    # 4) Determine the threshold as the specified percentile of gaps
    threshold = np.percentile(gaps, percentile)
    
    # 5) Find the largest gap
    max_gap_idx = np.argmax(gaps)
    max_gap_value = gaps[max_gap_idx]
    
    # 6) Compare the largest gap to the threshold
    if max_gap_value >= threshold and not np.all(gaps == max_gap_value):
        # Split the data at the largest gap
        left_group = data_sorted[:max_gap_idx + 1]
        right_group = data_sorted[max_gap_idx + 1:]
        
        # Recursively split each subgroup
        left_subgroups = percentile_gap_split_multidim(left_group, percentile, min_group_size)
        right_subgroups = percentile_gap_split_multidim(right_group, percentile, min_group_size)
        
        return left_subgroups + right_subgroups
    else:
        # No significant gap found; treat entire data as one group
        return [data_sorted]
