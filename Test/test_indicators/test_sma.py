import sys
import os

# 将项目根目录添加到 Python 模块搜索路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import numpy as np
import pandas as pd
import pandas_ta as ta  # 移到顶部导入

# 导入拆分后的工具函数
from indicators.sma import sma_njit, sma_name, sma_result_name, sma_load, sma_talib
from utils.constants import np_float
from indicators.indicators_config import create_params, get_value
from numba_calculation_api import calculate_jit_timer
from utils.data_loading import convert_to_pandas_dataframe
from Test.test_utils import (
    ohlcv_data_df,
    ohlcv_data_numpy,
    _assert_indicator_accuracy,
    get_column_index,
)


def test_sma_accuracy(ohlcv_data_df, ohlcv_data_numpy):
    """
    测试 SMA 指标的准确性，对比 pandas_ta 和 numba 实现。
    talib False 可通过
    talib True 可通过
    """
    print("\n------test_sma_accuracy------\n")
    period = 14
    close_series = ohlcv_data_df["close"]
    close_prices_numpy = ohlcv_data_numpy[:, get_column_index(ohlcv_data_df, "close")]

    # Pandas TA 计算 SMA
    pandas_sma = ta.sma(close=close_series, length=period, talib=sma_talib).values

    # Numba 实现计算 SMA
    numba_sma_result = np.full_like(close_prices_numpy, np.nan, dtype=np_float)
    sma_njit(close_prices_numpy, period, numba_sma_result)

    print("sma", numba_sma_result)
    _assert_indicator_accuracy(pandas_sma, numba_sma_result, "SMA", f"period={period}")


def test_sma_with_config_and_jit(
    ohlcv_data_df,
    ohlcv_data_numpy,
):
    """
    测试 SMA 指标，使用配置和 JIT 编译，并与 pandas-ta 进行比较。
    talib False 可通过
    talib True 可通过
    """
    print("\n------test_sma_with_config_and_jit------\n")
    test_config = [
        {
            "indicatros": [
                {
                    "active": True,
                    "name": sma_name,
                    "result_name": sma_result_name,
                    "params": [{"name": "period", "value": 14}],
                    "load": get_value(sma_load, 1),
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

    # 从配置中获取 SMA 的周期
    period = test_config[0]["indicatros"][0]["params"][0]["value"]

    # Pandas TA 计算 SMA
    close_series = ohlcv_data_df["close"]
    pandas_sma = ta.sma(close=close_series, length=period, talib=sma_talib).values

    # 提取 Numba 计算结果
    numba_sma_result = df_result[f"{sma_result_name[0][0]}_{period}"].values

    # print(df_result)

    # 比较结果
    _assert_indicator_accuracy(
        pandas_sma,
        numba_sma_result,
        "SMA (Config & JIT)",
        f"period={period}",
    )
