import sys
import copy
import os
import numpy as np
import pandas as pd
import pandas_ta as ta
import pytest

# Add the project root to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from indicators.sma import sma_name, sma_result_name, sma_load, sma_talib
from indicators.bbands import bbands_name, bbands_result_name, bbands_load, bbands_talib
from indicators.rsi import rsi_name, rsi_result_name, rsi_load, rsi_talib
from indicators.atr import atr_name, atr_result_name, atr_load, atr_talib
from indicators.psar import psar_name, psar_result_name, psar_load, psar_talib
from indicators.indicators_config import create_params, get_value
from numba_calculation_api import calculate_jit_timer
from utils.data_loading import convert_to_pandas_dataframe
from Test.test_utils import (
    ohlcv_data_df,
    ohlcv_data_numpy,
    _assert_indicator_accuracy,
    get_column_index,
)

test_configs = [
    {
        "indicatros": [
            {
                "active": True,
                "name": sma_name,
                "result_name": sma_result_name,
                "params": [{"name": "period", "value": 14}],
                "load": get_value(sma_load, 1),
            },
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
            {
                "active": True,
                "name": rsi_name,
                "result_name": rsi_result_name,
                "params": [{"name": "period", "value": 14}],
                "load": get_value(rsi_load, 1),
            },
            {
                "active": True,
                "name": atr_name,
                "result_name": atr_result_name,
                "params": [{"name": "period", "value": 14}],
                "load": get_value(atr_load, 1),
            },
            {
                "active": True,
                "name": psar_name,
                "result_name": psar_result_name,
                "params": [
                    {"name": "af0", "value": 0.02},
                    {"name": "af", "value": 0.02},
                    {"name": "max_af", "value": 0.2},
                ],
                "load": get_value(psar_load, 1),
            },
        ]
    }
]


def _test_indicator_config_accuracy(config, ohlcv_data_df, ohlcv_data_numpy):
    print(f"\n------_test_indicator_config_accuracy for config: {config}------\n")

    np_params, np_load, result_name, segmented_params = create_params(config)
    out_arrays = calculate_jit_timer(ohlcv_data_numpy, np_params, np_load)
    df_result = convert_to_pandas_dataframe(
        segmented_params[0], result_name, out_arrays[0]
    )

    for i, indicator_config in enumerate(config[0]["indicatros"]):
        if not indicator_config["active"]:
            print(f"Skipping inactive indicator: {indicator_config['name']}")
            continue

        indicator_name = indicator_config["name"]
        period = indicator_config["params"][0]["value"]

        pandas_result_list = []
        numba_result_list = []
        indicator_display_name = ""

        if indicator_name == sma_name:
            close_series = ohlcv_data_df["close"]
            pandas_result = ta.sma(
                close=close_series, length=period, talib=sma_talib
            ).values
            pandas_result_list.append(pandas_result)
            numba_result_list.append(
                df_result[f"{sma_result_name[0][0]}_{period}"].values
            )
            indicator_display_name = "SMA"
        elif indicator_name == bbands_name:
            close_series = ohlcv_data_df["close"]
            std_mult = indicator_config["params"][1]["value"]
            pandas_bbands = ta.bbands(
                close=close_series, length=period, std=std_mult, talib=bbands_talib
            )

            pandas_result_list.append(pandas_bbands[f"BBM_{period}_{std_mult}"].values)
            pandas_result_list.append(pandas_bbands[f"BBU_{period}_{std_mult}"].values)
            pandas_result_list.append(pandas_bbands[f"BBL_{period}_{std_mult}"].values)

            numba_result_list.append(
                df_result[f"{bbands_result_name[0][0]}_{period}_{std_mult}"].values
            )
            numba_result_list.append(
                df_result[f"{bbands_result_name[1][0]}_{period}_{std_mult}"].values
            )
            numba_result_list.append(
                df_result[f"{bbands_result_name[2][0]}_{period}_{std_mult}"].values
            )
            indicator_display_name = "BBands"
        elif indicator_name == rsi_name:
            close_series = ohlcv_data_df["close"]
            pandas_result = ta.rsi(
                close=close_series, length=period, talib=rsi_talib
            ).values
            pandas_result_list.append(pandas_result)
            numba_result_list.append(
                df_result[f"{rsi_result_name[0][0]}_{period}"].values
            )
            indicator_display_name = "RSI"
        elif indicator_name == atr_name:
            high_series = ohlcv_data_df["high"]
            low_series = ohlcv_data_df["low"]
            close_series = ohlcv_data_df["close"]
            pandas_result = ta.atr(
                high=high_series,
                low=low_series,
                close=close_series,
                length=period,
                talib=atr_talib,
            ).values
            pandas_result_list.append(pandas_result)
            numba_result_list.append(
                df_result[f"{atr_result_name[0][0]}_{period}"].values
            )
            indicator_display_name = "ATR"
        elif indicator_name == psar_name:
            af0 = indicator_config["params"][0]["value"]
            af = indicator_config["params"][1]["value"]
            max_af = indicator_config["params"][2]["value"]
            high_series = ohlcv_data_df["high"]
            low_series = ohlcv_data_df["low"]
            close_series = ohlcv_data_df["close"]
            pandas_psar = ta.psar(
                high=high_series,
                low=low_series,
                close=close_series,
                af0=af0,
                af=af,
                max_af=max_af,
                talib=psar_talib,
            )
            pandas_result_list.append(pandas_psar[f"PSARl_{af0}_{max_af}"].values)
            pandas_result_list.append(pandas_psar[f"PSARs_{af0}_{max_af}"].values)
            numba_result_list.append(
                df_result[f"{psar_result_name[0][0]}_{af0}_{af}_{max_af}"].values
            )
            numba_result_list.append(
                df_result[f"{psar_result_name[1][0]}_{af0}_{af}_{max_af}"].values
            )
            indicator_display_name = "PSAR"
        else:
            raise ValueError(f"Unknown indicator: {indicator_name}")

        for j, (pandas_res, numba_res) in enumerate(
            zip(pandas_result_list, numba_result_list)
        ):
            sub_indicator_name = (
                indicator_config["result_name"][j][0]
                if len(indicator_config["result_name"]) > 1
                else ""
            )
            full_display_name = f"{indicator_display_name} {sub_indicator_name}".strip()
            _assert_indicator_accuracy(
                pandas_res,
                numba_res,
                f"{full_display_name} (Config & JIT)",
                f"period={period}",
            )


def test_all_indicators_with_config_and_jit_original(ohlcv_data_df, ohlcv_data_numpy):
    """
    测试所有硬编码指标的准确性，使用原始配置和 JIT 编译，并与 pandas-ta 进行比较。
    """
    config = copy.deepcopy(test_configs)
    _test_indicator_config_accuracy(config, ohlcv_data_df, ohlcv_data_numpy)


def test_all_indicators_with_config_and_jit_scenario1(ohlcv_data_df, ohlcv_data_numpy):
    """
    测试所有硬编码指标的准确性，场景一：隐藏第二个指标。
    """
    config = copy.deepcopy(test_configs)
    if len(config[0]["indicatros"]) > 1:
        config[0]["indicatros"][1]["active"] = False
    _test_indicator_config_accuracy(config, ohlcv_data_df, ohlcv_data_numpy)


def test_all_indicators_with_config_and_jit_scenario2(ohlcv_data_df, ohlcv_data_numpy):
    """
    测试所有硬编码指标的准确性，场景二：调换指标顺序并隐藏第三个指标。
    """
    config = copy.deepcopy(test_configs)
    indicators = config[0]["indicatros"]
    if len(indicators) >= 4:
        # 调换第二个和第四个指标
        indicators[1], indicators[3] = indicators[3], indicators[1]
    if len(indicators) >= 3:
        # 隐藏第三个指标 (原第三个，即调换后的新第三个)
        indicators[2]["active"] = False
    _test_indicator_config_accuracy(config, ohlcv_data_df, ohlcv_data_numpy)
