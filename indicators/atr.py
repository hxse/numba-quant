import numpy as np
from numba import types
from utils.numba_helpers import numba_factory
from utils.constants import numba_int, numba_float
from .tr import tr_jit, tr_njit, tr_cuda
from .rma import rma_jit, rma_njit, rma_cuda

atr_name = "atr"
atr_result_name = [["atr", 1], ["tr", 0]]  # 1指标名,2是否显示到dataframe中
atr_load = [
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
        2,
        # atr_result, tr_result
    ],  # 表明该指标会输入多少列数据，直接决定了其输入的宽度。
    [
        "indicator_id",
        3,
    ],  # 代表该指标的id,不能跟其他指标重复。
]
atr_talib = True


def get_atr_logic(tr_func, rma_func):
    """
    返回平均真实范围 (ATR) 的核心 Python 逻辑。
    接收已编译的 TR 和 RMA 函数，填充 ATR 和 TR 数组。
    """

    def _actual_atr_logic(high, low, close, period, atr_result, tr_result):
        n = len(high)

        # 这里不需要初始化 result 为 NaN，因为 rma_func 应该负责填充
        # tr_result 也由 tr_func 填充，所以这里的初始化也可以移除，但为了安全保留
        atr_result[:] = np.nan
        tr_result[:] = np.nan

        # 计算真实范围 (TR) 值
        # tr_func 会将 TR 值填充到 tr_result 数组中，并在 tr_result[0] 处留下 NaN
        tr_func(high, low, close, tr_result)

        # 计算真实范围的移动平均 (RMA)
        # rma_func 应该将 RMA 值填充到 result 数组中。
        # RMA 应该在 period - 1 个 NaN 之后开始计算有效值。
        rma_func(tr_result, period, atr_result)

        # 移除这个循环！
        # 这个循环导致了多一个 NaN，并且如果你之前的 RMA 已经正确处理了 NaN 数量
        # 那么这个循环会把第一个有效值也覆盖掉。
        # 正确的 RMA 实现本身就应该在第 period 个索引处开始输出有效值。
        # for i in range(min(period, n)):
        #     result[i] = np.nan

        return (atr_result, tr_result)

    return _actual_atr_logic


def create_atr_function(mode, tr_func_compiled, rma_func_compiled):
    """
    创建并编译一个 Numba JIT 函数，用于计算平均真实范围 (ATR)。

    参数:
        mode (str): Numba 编译模式 ("jit", "njit", "cuda")。
        tr_func_compiled (function): 已编译的 TR 函数。
        rma_func_compiled (function): 已编译的 RMA 函数。

    返回:
        function: 一个 Numba JIT 编译的函数，用于计算 ATR。
    """
    _atr_py_logic = get_atr_logic(tr_func_compiled, rma_func_compiled)

    atr_func_compiled = numba_factory(
        mode,
        types.Tuple((numba_float[:], numba_float[:]))(
            numba_float[:],
            numba_float[:],
            numba_float[:],
            numba_int,
            numba_float[:],
            numba_float[:],
        ),
    )(_atr_py_logic)

    return atr_func_compiled


atr_jit = create_atr_function("jit", tr_jit, rma_jit)
atr_njit = create_atr_function("njit", tr_njit, rma_njit)
atr_cuda = create_atr_function("cuda", tr_cuda, rma_cuda)
