import sys
import os
import pytest
import numpy as np  # 添加 numpy 导入
from utils.data_loading import load_ohlcv_from_csv, convert_ohlcv_numpy
from utils.numpy_utils import get_leading_nan_counts_for_two_arrays  # 添加导入

# 将项目根目录添加到 Python 模块搜索路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 全局数据配置
DATA_SIZE = 1000 * 200
CSV_FILE_PATH = "csv/BTC_USDT/future live/4h/2022-01-01 00_00_00.csv"

GLOBAL_RTOL = 1e-5
GLOBAL_ATOL = 1e-8


@pytest.fixture(scope="module")
def ohlcv_data_df():
    """
    Fixture to load OHLCV data once for all tests in the module.
    """
    df = load_ohlcv_from_csv(CSV_FILE_PATH, DATA_SIZE)
    return df


@pytest.fixture(scope="module")
def ohlcv_data_numpy(ohlcv_data_df):
    """
    Fixture to convert OHLCV DataFrame to NumPy array.
    """
    return convert_ohlcv_numpy(ohlcv_data_df)


def _assert_indicator_accuracy(pandas_result, numba_result, indicator_name, params_str):
    """
    通用函数，用于比较 Pandas TA 和 Numba 实现的指标结果。
    """
    print("\n")
    valid_indices = ~np.isnan(pandas_result) & ~np.isnan(numba_result)

    pandas_nan_count, numba_nan_count = get_leading_nan_counts_for_two_arrays(
        pandas_result, numba_result
    )
    print(
        f"{indicator_name} ({params_str}) pandas_nan_count: {pandas_nan_count} numba_nan_count: {numba_nan_count}"
    )
    assert pandas_nan_count == numba_nan_count, (
        f"{indicator_name} leading NaN count mismatch: Pandas has {pandas_nan_count}, Numba has {numba_nan_count}"
    )

    # 计算并打印最大差值，只考虑有效索引
    max_diff = (
        np.max(np.abs(pandas_result[valid_indices] - numba_result[valid_indices]))
        if np.any(valid_indices)
        else 0.0
    )
    print(f"{indicator_name} ({params_str}) - Max difference: {max_diff:.4e}")

    np.testing.assert_allclose(
        pandas_result[valid_indices],
        numba_result[valid_indices],
        rtol=GLOBAL_RTOL,
        atol=GLOBAL_ATOL,
        err_msg=f"{indicator_name} calculation mismatch for {params_str}",
    )
    print(f"{indicator_name} ({params_str}) accuracy test passed.")


def get_column_index(ohlcv_data_df, column_name):
    """
    获取 OHLCV DataFrame 中指定列的索引。
    """
    return ohlcv_data_df.columns.get_loc(column_name)
