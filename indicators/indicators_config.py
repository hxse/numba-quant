from .sma import sma_name, sma_result_name, sma_load, sma_name
from .bbands import (
    bbands_name,
    bbands_result_name,
    bbands_load,
    bbands_name,
)
from .rsi import rsi_name, rsi_result_name, rsi_load, rsi_name
from .atr import atr_name, atr_result_name, atr_load, atr_name
from .psar import psar_name, psar_result_name, psar_load
import numpy as np
from utils.constants import ENABLE_CACHE, np_int, np_float, numba_int, numba_float
from utils.indicator_processing import (
    update_indicator_load_cumulative,
    extract_segmented_params,
    extract_segmented_2d_array_by_load_params,
)


def get_value(array, idx):
    return [i[idx] for i in array]


def sync_active_status(config):
    """
    config 第一行的active,去覆盖后续行对应的active
    config 长度需要对齐
    """
    if not config or len(config) < 2:
        return config

    source_indicators = config[0]["indicatros"]
    source_len = len(source_indicators)

    for i in range(1, len(config)):
        target_indicators = config[i]["indicatros"]
        assert len(target_indicators) == source_len, (
            f"Config block at index {i} has a different number of indicators than the first block."
        )

        for j in range(source_len):
            assert source_indicators[j]["name"] == target_indicators[j]["name"], (
                f"Indicator name mismatch at index {j} in config block {i}. Expected '{source_indicators[j]['name']}', got '{target_indicators[j]['name']}'."
            )
            target_indicators[j]["active"] = source_indicators[j]["active"]
    return config


def create_params(config):
    config = sync_active_status(config)

    np_params = np.array(
        [
            [p["value"] for i in _c["indicatros"] if i["active"] for p in i["params"]]
            for _c in config
        ],
        dtype=np_float,
    )

    np_load = np.array(
        [i["load"] for i in config[0]["indicatros"] if i["active"]],
        dtype=np_int,
    )

    result_name = [i["result_name"] for i in config[0]["indicatros"] if i["active"]]

    np_load = update_indicator_load_cumulative(np_load, 0, 1)
    np_load = update_indicator_load_cumulative(np_load, 2, 3)

    assert np_load.shape[0] == len(result_name), (
        "错误: np_load 的行数与 result_name 的长度不匹配。"
    )
    for i in range(np_load.shape[0]):
        assert np_load[i, 3] == len(result_name[i]), (
            f"错误: np_load 第 {i} 行的第四列 ({np_load[i, 3]}) "
            f"与 result_name 第 {i} 个子数组的长度 ({len(result_name[i])}) 不匹配。"
        )
    assert np.sum(np_load[:, 1]) == np_params.shape[1], (
        "错误: np_load 第二列的累积和与 np_params 的列数不匹配。"
    )

    segmented_params = extract_segmented_2d_array_by_load_params(np_params, np_load)
    return np_params, np_load, result_name, segmented_params


config = [
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
                "name": atr_name,
                "result_name": atr_result_name,
                "params": [{"name": "period", "value": 14}],
                "load": get_value(atr_load, 1),
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
                "name": psar_name,
                "result_name": psar_result_name,
                "params": [
                    {"name": "af0", "value": 0.02},
                    {"name": "af", "value": 0.02},
                    {"name": "max_af", "value": 0.2},
                ],
                "load": get_value(psar_load, 1),
            },
            {
                "active": True,
                "name": sma_name,
                "result_name": sma_result_name,
                "params": [{"name": "period", "value": 20}],
                "load": get_value(sma_load, 1),
            },
        ]
    },
    {
        "indicatros": [
            {
                "name": sma_name,
                "result_name": sma_result_name,
                "params": [{"name": "period", "value": 20}],
                "load": get_value(sma_load, 1),
            },
            {
                "name": atr_name,
                "result_name": atr_result_name,
                "params": [{"name": "period", "value": 20}],
                "load": get_value(atr_load, 1),
            },
            {
                "name": bbands_name,
                "result_name": bbands_result_name,
                "params": [
                    {"name": "period", "value": 20},
                    {"name": "std_mult", "value": 2},
                ],
                "load": get_value(bbands_load, 1),
            },
            {
                "name": rsi_name,
                "result_name": rsi_result_name,
                "params": [{"name": "period", "value": 20}],
                "load": get_value(rsi_load, 1),
            },
            {
                "name": psar_name,
                "result_name": psar_result_name,
                "params": [
                    {"name": "af0", "value": 0.02},
                    {"name": "af", "value": 0.02},
                    {"name": "max_af", "value": 0.2},
                ],
                "load": get_value(psar_load, 1),
            },
            {
                "name": sma_name,
                "result_name": sma_result_name,
                "params": [{"name": "period", "value": 50}],
                "load": get_value(sma_load, 1),
            },
        ]
    },
]
