import sys
import os

# 将项目根目录添加到 Python 模块搜索路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import numpy as np
import pandas as pd
import pandas_ta as ta

# 导入拆分后的工具函数
from indicators.psar import (
    psar_njit,
    psar_result_name,
    psar_name,
    psar_load,
    psar_talib,
)
from indicators.indicators_config import create_params, get_value
from numba_calculation_api import calculate_jit_timer
from utils.data_loading import convert_to_pandas_dataframe
from utils.constants import np_float
from Test.test_utils import (
    ohlcv_data_df,
    ohlcv_data_numpy,
    _assert_indicator_accuracy,
    get_column_index,
)


def test_psar_accuracy(ohlcv_data_df, ohlcv_data_numpy):
    """
    测试 PSAR 指标的准确性，对比 pandas_ta 和 numba 实现。
    talib False 可通过
    talib True 可通过
    """

    print("\n------test_psar_accuracy------\n")
    af0 = 0.02
    af = 0.02
    max_af = 0.2

    high_prices_df = ohlcv_data_df["high"]
    low_prices_df = ohlcv_data_df["low"]
    close_prices_df = ohlcv_data_df["close"]

    high_prices_numpy = ohlcv_data_numpy[:, get_column_index(ohlcv_data_df, "high")]
    low_prices_numpy = ohlcv_data_numpy[:, get_column_index(ohlcv_data_df, "low")]
    close_prices_numpy = ohlcv_data_numpy[:, get_column_index(ohlcv_data_df, "close")]

    # Pandas TA 计算 PSAR
    pandas_psar = ta.psar(
        high=high_prices_df,
        low=low_prices_df,
        close=close_prices_df,
        af0=af0,
        af=af,
        max_af=max_af,
        talib=psar_talib,
    )
    # pandas_ta 的 PSAR 返回 PSARl (Long) 和 PSARs (Short)
    pandas_psar_long = pandas_psar[f"PSARl_{af0}_{max_af}"].values
    pandas_psar_short = pandas_psar[f"PSARs_{af0}_{max_af}"].values

    # Numba 实现计算 PSAR
    numba_psar_long_result = np.full_like(close_prices_numpy, np.nan, dtype=np_float)
    numba_psar_short_result = np.full_like(close_prices_numpy, np.nan, dtype=np_float)
    numba_psar_af_result = np.full_like(close_prices_numpy, np.nan, dtype=np_float)
    numba_psar_reversal_result = np.full_like(
        close_prices_numpy, np.nan, dtype=np_float
    )

    (
        numba_psar_long_result,
        numba_psar_short_result,
        numba_psar_af_result,
        numba_psar_reversal_result,
    ) = psar_njit(
        high_prices_numpy,
        low_prices_numpy,
        close_prices_numpy,
        af0,
        af,
        max_af,
        numba_psar_long_result,
        numba_psar_short_result,
        numba_psar_af_result,
        numba_psar_reversal_result,
    )

    print("PSAR Long", numba_psar_long_result)
    print("PSAR Short", numba_psar_short_result)
    print("PSAR AF", numba_psar_af_result)
    print("PSAR Reversal", numba_psar_reversal_result[:10])
    print(pandas_psar)

    _assert_indicator_accuracy(
        pandas_psar_long,
        numba_psar_long_result,
        "PSAR Long",
        f"af0={af0}, af={af}, max_af={max_af}",
    )
    _assert_indicator_accuracy(
        pandas_psar_short,
        numba_psar_short_result,
        "PSAR Short",
        f"af0={af0}, af={af}, max_af={max_af}",
    )


def test_psar_with_config_and_jit(ohlcv_data_df, ohlcv_data_numpy):
    """
    测试 PSAR 指标，使用配置和 JIT 编译，并与 pandas-ta 进行比较。
    talib False 可通过
    talib True 可通过
    """

    print("\n------test_psar_with_config_and_jit------\n")
    af0 = 0.02
    af = 0.02
    max_af = 0.2

    test_config = [
        {
            "indicatros": [
                {
                    "active": True,
                    "name": psar_name,
                    "result_name": psar_result_name,
                    "params": [
                        {"name": "af0", "value": af0},
                        {"name": "af", "value": af},
                        {"name": "max_af", "value": max_af},
                    ],
                    "load": get_value(psar_load, 1),
                },
            ]
        }
    ]

    # 使用精简后的配置创建参数
    np_params, np_load, result_name, segmented_params = create_params(test_config)

    # 使用 calculate_jit_timer 计算指标
    out_arrays = calculate_jit_timer(ohlcv_data_numpy, np_params, np_load)

    # 将结果转换为 Pandas DataFrame
    df_result = convert_to_pandas_dataframe(
        segmented_params[0], result_name, out_arrays[0]
    )

    # Pandas TA 计算 PSAR
    high_prices_df = ohlcv_data_df["high"]
    low_prices_df = ohlcv_data_df["low"]
    close_prices_df = ohlcv_data_df["close"]
    pandas_psar = ta.psar(
        high=high_prices_df,
        low=low_prices_df,
        close=close_prices_df,
        af0=af0,
        af=af,
        max_af=max_af,
        talib=psar_talib,
    )
    pandas_psar_long = pandas_psar[f"PSARl_{af0}_{max_af}"].values
    pandas_psar_short = pandas_psar[f"PSARs_{af0}_{max_af}"].values

    # 提取 Numba 计算结果
    numba_psar_long_result = df_result[
        f"{psar_result_name[0][0]}_{af0}_{af}_{max_af}"
    ].values
    numba_psar_short_result = df_result[
        f"{psar_result_name[1][0]}_{af0}_{af}_{max_af}"
    ].values
    # numba_psar_af_result = df_result[f"{psar_result_name[2][0]}_{af0}_{max_af}"].values
    # numba_psar_reversal_result = df_result[f"{psar_result_name[3][0]}_{af0}_{max_af}"].values

    print(df_result)

    # 比较结果
    _assert_indicator_accuracy(
        pandas_psar_long,
        numba_psar_long_result,
        "PSAR Long (Config & JIT)",
        f"af0={af0}, af={af}, max_af={max_af}",
    )
    _assert_indicator_accuracy(
        pandas_psar_short,
        numba_psar_short_result,
        "PSAR Short (Config & JIT)",
        f"af0={af0}, af={af}, max_af={max_af}",
    )
