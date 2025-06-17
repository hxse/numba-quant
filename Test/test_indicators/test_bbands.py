import sys
import os

# 将项目根目录添加到 Python 模块搜索路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import numpy as np
import pandas as pd
import pandas_ta as ta

# 导入拆分后的工具函数
from indicators.bbands import (
    bollinger_njit,
    bbands_result_name,
    bbands_name,
    bbands_load,
    bbands_talib,
)  # 使用njit版本
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


def test_bollinger_bands_accuracy(ohlcv_data_df, ohlcv_data_numpy):
    """
    测试布林带指标的准确性，对比 pandas_ta 和 numba 实现。
    talib False 可通过
    talib True 可通过
    """
    print("\n------test_bollinger_bands_accuracy------\n")
    period = 14
    std_mult = 2.5
    close_series = ohlcv_data_df["close"]
    close_prices_numpy = ohlcv_data_numpy[:, get_column_index(ohlcv_data_df, "close")]

    # Pandas TA 计算布林带
    pandas_bbands = ta.bbands(
        close=close_series, length=period, std=std_mult, talib=bbands_talib
    )
    pandas_middle = pandas_bbands[f"BBM_{period}_{std_mult}"].values
    pandas_upper = pandas_bbands[f"BBU_{period}_{std_mult}"].values
    pandas_lower = pandas_bbands[f"BBL_{period}_{std_mult}"].values

    # Numba 实现计算布林带
    numba_middle_result = np.full_like(close_prices_numpy, np.nan, dtype=np_float)
    numba_upper_result = np.full_like(close_prices_numpy, np.nan, dtype=np_float)
    numba_lower_result = np.full_like(close_prices_numpy, np.nan, dtype=np_float)

    bollinger_njit(
        close_prices_numpy,
        period,
        std_mult,
        numba_middle_result,
        numba_upper_result,
        numba_lower_result,
    )

    print("BBands Middle Band", numba_middle_result)
    print("BBands Upper Band", numba_upper_result)
    print("BBands Lower Band", numba_lower_result)

    _assert_indicator_accuracy(
        pandas_middle,
        numba_middle_result,
        "BBands Middle Band",
        f"period={period}, std_mult={std_mult}",
    )
    _assert_indicator_accuracy(
        pandas_upper,
        numba_upper_result,
        "BBands Upper Band",
        f"period={period}, std_mult={std_mult}",
    )
    _assert_indicator_accuracy(
        pandas_lower,
        numba_lower_result,
        "BBands Lower Band",
        f"period={period}, std_mult={std_mult}",
    )


def test_bollinger_bands_with_config_and_jit(ohlcv_data_df, ohlcv_data_numpy):
    """
    测试布林带指标，使用配置和 JIT 编译，并与 pandas-ta 进行比较。
    talib False 可通过
    talib True 可通过
    """
    print("\n------test_bollinger_bands_with_config_and_jit------\n")
    test_config = [
        {
            "indicatros": [
                {
                    "active": True,
                    "name": bbands_name,
                    "result_name": bbands_result_name,
                    "params": [
                        {"name": "period", "value": 14},
                        {"name": "std_mult", "value": 2.5},
                    ],
                    "load": get_value(bbands_load, 1),
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

    # 从配置中获取布林带的周期和标准差乘数
    period = test_config[0]["indicatros"][0]["params"][0]["value"]
    std_mult = test_config[0]["indicatros"][0]["params"][1]["value"]

    # Pandas TA 计算布林带
    close_series = ohlcv_data_df["close"]
    pandas_bbands = ta.bbands(
        close=close_series, length=period, std=std_mult, talib=bbands_talib
    )

    name0 = f"BBM_{period}_{std_mult}"
    name1 = f"BBU_{period}_{std_mult}"
    name2 = f"BBL_{period}_{std_mult}"
    pandas_middle = pandas_bbands[name0].values
    pandas_upper = pandas_bbands[name1].values
    pandas_lower = pandas_bbands[name2].values

    name0 = f"{bbands_result_name[0][0]}_{period}_{std_mult}"
    name1 = f"{bbands_result_name[1][0]}_{period}_{std_mult}"
    name2 = f"{bbands_result_name[2][0]}_{period}_{std_mult}"
    # 提取 Numba 计算结果
    numba_middle_result = df_result[name0].values
    numba_upper_result = df_result[name1].values
    numba_lower_result = df_result[name2].values

    # print(df_result)

    # 比较结果
    _assert_indicator_accuracy(
        pandas_middle,
        numba_middle_result,
        "BBands Middle Band (Config & JIT)",
        f"period={period}, std_mult={std_mult}",
    )
    _assert_indicator_accuracy(
        pandas_upper,
        numba_upper_result,
        "BBands Upper Band (Config & JIT)",
        f"period={period}, std_mult={std_mult}",
    )
    _assert_indicator_accuracy(
        pandas_lower,
        numba_lower_result,
        "BBands Lower Band (Config & JIT)",
        f"period={period}, std_mult={std_mult}",
    )
