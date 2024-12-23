import unittest
import numpy as np
from adaptive_gap_splitter import percentile_gap_split_1d, percentile_gap_split_multidim

class TestAdaptiveGapSplitter(unittest.TestCase):
    
    def test_splitter_1d_no_split(self):
        data = [1, 2, 3, 4, 5]
        groups = percentile_gap_split_1d(data, percentile=95)
        self.assertEqual(len(groups), 1)
        np.testing.assert_array_equal(groups[0], np.array([1, 2, 3, 4, 5]))
    
    def test_splitter_1d_single_split(self):
        data = [1, 2, 3, 4, 5, 10, 11, 12]
        groups = percentile_gap_split_1d(data, percentile=90)
        self.assertEqual(len(groups), 2)
        np.testing.assert_array_equal(groups[0], np.array([1, 2, 3, 4, 5]))
        np.testing.assert_array_equal(groups[1], np.array([10, 11, 12]))
    
    def test_splitter_1d_multiple_splits(self):
        data = [1, 2, 3, 10, 11, 20, 21, 30]
        groups = percentile_gap_split_1d(data, percentile=90)
        self.assertEqual(len(groups), 4)
        expected_groups = [
            np.array([1, 2, 3]),
            np.array([10, 11]),
            np.array([20, 21]),
            np.array([30])
        ]
        for group, expected in zip(groups, expected_groups):
            np.testing.assert_array_equal(group, expected)
    
    def test_splitter_multidim_no_split(self):
        data = [
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5],
            [5, 6]
        ]
        groups = percentile_gap_split_multidim(data, percentile=95)
        self.assertEqual(len(groups), 1)
        np.testing.assert_array_equal(groups[0], np.array(data))
    
    def test_splitter_multidim_single_split(self):
        data = [
            [1, 2],
            [2, 3],
            [3, 4],
            [10, 10],
            [11, 11],
            [12, 12]
        ]
        groups = percentile_gap_split_multidim(data, percentile=90)
        self.assertEqual(len(groups), 2)
        expected_groups = [
            np.array([
                [1, 2],
                [2, 3],
                [3, 4]
            ]),
            np.array([
                [10, 10],
                [11, 11],
                [12, 12]
            ])
        ]
        for group, expected in zip(groups, expected_groups):
            np.testing.assert_array_equal(group, expected)
            
    def test_splitter_1d_all_gaps_equal_threshold(self):
        data = [1, 2, 3, 4, 5]
        groups = percentile_gap_split_1d(data, percentile=100)
        self.assertEqual(len(groups), 1)
        np.testing.assert_array_equal(groups[0], np.array([1, 2, 3, 4, 5]))


    def test_splitter_1d_multiple_equal_largest_gaps(self):
        data = [1, 2, 3, 10, 11, 20, 21, 30]
        groups = percentile_gap_split_1d(data, percentile=90)
        self.assertEqual(len(groups), 4)
        expected_groups = [
            np.array([1, 2, 3]),
            np.array([10, 11]),
            np.array([20, 21]),
            np.array([30])
        ]
        for group, expected in zip(groups, expected_groups):
            np.testing.assert_array_equal(group, expected)

if __name__ == '__main__':
    unittest.main()
