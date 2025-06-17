import numpy as np
from typing import Tuple


def _count_leading_nans(arr: np.ndarray) -> int:
    """
    计算 NumPy 数组开头连续 NaN 值的数量。

    Args:
        arr (np.ndarray): 输入的 NumPy 数组。

    Returns:
        int: 开头连续 NaN 值的数量。
    """
    if arr.size == 0:
        return 0

    # 找到所有非 NaN 值的索引
    non_nan_indices = np.where(~np.isnan(arr))[0]

    if non_nan_indices.size == 0:
        # 如果没有非 NaN 值，则所有元素都是 NaN
        return arr.size
    else:
        # 第一个非 NaN 值的索引就是前导 NaN 的数量
        return non_nan_indices[0]


def get_leading_nan_counts_for_two_arrays(
    arr1: np.ndarray, arr2: np.ndarray
) -> Tuple[int, int]:
    """
    计算两个 NumPy 数组开头 NaN 值的数量。

    Args:
        arr1 (np.ndarray): 第一个 NumPy 数组。
        arr2 (np.ndarray): 第二个 NumPy 数组。

    Returns:
        Tuple[int, int]: 一个元组，包含 arr1 和 arr2 开头 NaN 值的数量。
    """
    nan_count_arr1 = _count_leading_nans(arr1)
    nan_count_arr2 = _count_leading_nans(arr2)
    return nan_count_arr1, nan_count_arr2
