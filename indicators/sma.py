import math
import numpy as np
from numba import types
from utils.numba_helpers import numba_factory
from utils.constants import numba_int, numba_float

sma_name = "sma"
sma_result_name = [["sma", 1]]  # 1指标名,2是否显示到dataframe中
sma_load = [
    [
        "param_idx",
        -1,
    ],  # 定义了该指标的参数在总参数池中的起始位置，用于精确查找所需参数。
    [
        "param_count",
        1,
        # period
    ],  # 指定了该指标需要多少个参数，决定了从参数池中提取数据的长度。
    [
        "col_idx",
        -1,
    ],  # 作为数据起始位置的占位符，通常用 -1 填充，用于指示指标数据在整体数据中的起始列。
    [
        "col_count",
        1,
        # sma_result
    ],  # 表明该指标会输入多少列数据，直接决定了其输入的宽度。
    [
        "indicator_id",
        0,
    ],  # 代表该指标的id,不能跟其他指标重复。
]
sma_talib = False


def get_sma_logic():
    """
    返回简单移动平均 (SMA) 的核心 Python 逻辑。
    计算整个数据数组的 SMA，填充到结果数组。
    """

    def _actual_sma_logic(close, period, sma_result):
        # 确保结果数组的长度与输入数据数组相同
        # 对于前 period - 1 个元素，填充 np.nan
        for i in range(min(period - 1, len(close))):  # 避免数据长度不足越界
            sma_result[i] = np.nan

        # 从 period - 1 索引开始计算 SMA
        for i in range(len(close) - period + 1):
            sum_val = 0.0
            for j in range(period):
                sum_val += close[i + j]
            sma_result[i + period - 1] = sum_val / period

        return sma_result

    return _actual_sma_logic


# 创建 SMA 函数
def create_sma_function(mode):
    _sma_py_logic = get_sma_logic()
    sma_func_compiled = numba_factory(
        mode, numba_float[:](numba_float[:], numba_int, numba_float[:])
    )(_sma_py_logic)
    return sma_func_compiled


# 为不同模式生成核心函数
sma_jit = create_sma_function("jit")
sma_njit = create_sma_function("njit")
sma_cuda = create_sma_function("cuda")
