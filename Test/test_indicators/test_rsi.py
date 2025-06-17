import sys
import os

# 将项目根目录添加到 Python 模块搜索路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import numpy as np
import pandas_ta as ta
import pandas as pd

from indicators.rsi import rsi_name, rsi_result_name, rsi_load, rsi_talib
from indicators.indicators_config import create_params, get_value
from numba_calculation_api import calculate_jit_timer
from utils.data_loading import convert_to_pandas_dataframe

# 导入拆分后的工具函数
from indicators.rsi import rsi_njit  # 导入rsi_njit
from utils.constants import np_float
from Test.test_utils import (
    ohlcv_data_df,
    ohlcv_data_numpy,
    _assert_indicator_accuracy,
    get_column_index,
)


def test_rsi_accuracy(ohlcv_data_df, ohlcv_data_numpy):
    """
    测试 RSI 指标的准确性，对比 pandas_ta 和 numba 实现。
    talib False 不能通过,前导nan数量不同,以talib True为准吧
    talib True 可通过
    """
    period = 14
    close_prices_df = ohlcv_data_df["close"]
    close_prices_numpy = ohlcv_data_numpy[:, get_column_index(ohlcv_data_df, "close")]

    # Pandas TA 计算 RSI
    pandas_rsi = ta.rsi(close=close_prices_df, length=period, talib=rsi_talib).values

    # Numba 实现计算 RSI
    numba_rsi_result = np.full_like(close_prices_numpy, np.nan, dtype=np_float)
    gains_result = np.full_like(close_prices_numpy, np.nan, dtype=np_float)
    losses_result = np.full_like(close_prices_numpy, np.nan, dtype=np_float)
    avg_gains_result = np.full_like(close_prices_numpy, np.nan, dtype=np_float)
    avg_losses_result = np.full_like(close_prices_numpy, np.nan, dtype=np_float)

    (
        numba_rsi_result,
        _,
        _,
        _,
        _,
    ) = rsi_njit(
        close_prices_numpy,
        period,
        numba_rsi_result,
        gains_result,
        losses_result,
        avg_gains_result,
        avg_losses_result,
    )

    _assert_indicator_accuracy(pandas_rsi, numba_rsi_result, "RSI", f"period={period}")


def test_rsi_with_config_and_jit(ohlcv_data_df, ohlcv_data_numpy):
    """
    测试 RSI 指标，使用配置和 JIT 编译，并与 pandas-ta 进行比较。
    talib False 不能通过,前导nan数量不同,以talib True为准吧
    talib True 可通过
    """
    print("\n------test_rsi_with_config_and_jit------\n")
    test_config = [
        {
            "indicatros": [
                {
                    "active": True,
                    "name": rsi_name,
                    "result_name": rsi_result_name,
                    "params": [{"name": "period", "value": 14}],
                    "load": get_value(rsi_load, 1),
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

    # 从配置中获取 RSI 的周期
    period = test_config[0]["indicatros"][0]["params"][0]["value"]

    # Pandas TA 计算 RSI
    close_series = ohlcv_data_df["close"]
    pandas_rsi = ta.rsi(close=close_series, length=period, talib=rsi_talib).values

    # 提取 Numba 计算结果
    numba_rsi_result = df_result[f"{rsi_result_name[0][0]}_{period}"].values

    # print(df_result)

    # 比较结果
    _assert_indicator_accuracy(
        pandas_rsi,
        numba_rsi_result,
        "RSI (Config & JIT)",
        f"period={period}",
    )
