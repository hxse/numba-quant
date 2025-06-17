import math
import numpy as np
from numba import types
from utils.numba_helpers import numba_factory
from .sma import sma_jit, sma_njit, sma_cuda
from utils.constants import ENABLE_CACHE, np_int, np_float, numba_int, numba_float

bbands_name = "bbands"
bbands_result_name = [
    ["bbm", 1],
    ["bbu", 1],
    ["bbl", 1],
]  # 1指标名,2是否显示到dataframe中
bbands_load = [
    [
        "param_idx",
        -1,
    ],  # 定义了该指标的参数在总参数池中的起始位置，用于精确查找所需参数。
    [
        "param_count",
        2,
        # period, std_mult
    ],  # 指定了该指标需要多少个参数，决定了从参数池中提取数据的长度。
    [
        "col_idx",
        -1,
    ],  # 作为数据起始位置的占位符，通常用 -1 填充，用于指示指标数据在整体数据中的起始列。
    [
        "col_count",
        3,
        # middle_result, upper_result, lower_result
    ],  # 表明该指标会输入多少列数据，直接决定了其输入的宽度。
    [
        "indicator_id",
        1,
    ],  # 代表该指标的id,不能跟其他指标重复。
]
bbands_talib = False


def get_bollinger_bands_logic(sma_func):
    """
    返回布林带计算的核心 Python 逻辑。
    接收已编译的 SMA 函数，填充中轨 (SMA)、上轨和下轨数组。
    """

    def _actual_bollinger_bands_logic(
        close, period, std_mult, middle_result, upper_result, lower_result
    ):
        # 计算 SMA 数组 (由 get_sma_logic 负责填充 np.nan)
        sma_func(close, period, middle_result)

        # 对于前 period - 1 个元素，填充 np.nan
        for i in range(min(period - 1, len(close))):  # 避免数据长度不足越界
            upper_result[i] = np.nan
            lower_result[i] = np.nan

        # 从 period - 1 索引开始计算标准差和布林带
        for i in range(len(close) - period + 1):
            variance = 0.0
            # 注意：这里 i + period - 1 是当前 SMA 对应的原始数据索引
            for j in range(period):
                diff = close[i + j] - middle_result[i + period - 1]
                variance += diff * diff
            std = math.sqrt(variance / period)
            upper_result[i + period - 1] = (
                middle_result[i + period - 1] + std_mult * std
            )
            lower_result[i + period - 1] = (
                middle_result[i + period - 1] - std_mult * std
            )

        return middle_result, upper_result, lower_result

    return _actual_bollinger_bands_logic


# 创建布林带函数，复用已编译的 SMA 函数
def create_bollinger_function(mode, sma_func_compiled):
    _bollinger_bands_py_logic = get_bollinger_bands_logic(sma_func_compiled)

    bollinger_bands_compiled = numba_factory(
        mode,
        types.Tuple((numba_float[:], numba_float[:], numba_float[:]))(
            numba_float[:],
            numba_int,
            numba_float,
            numba_float[:],
            numba_float[:],
            numba_float[:],
        ),
    )(_bollinger_bands_py_logic)

    return bollinger_bands_compiled


# 为不同模式生成核心函数
bollinger_jit = create_bollinger_function("jit", sma_jit)
bollinger_njit = create_bollinger_function("njit", sma_njit)
bollinger_cuda = create_bollinger_function("cuda", sma_cuda)
