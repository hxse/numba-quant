import numpy as np
from numba import types
from utils.numba_helpers import numba_factory
from utils.constants import ENABLE_CACHE, np_int, np_float, numba_int, numba_float
from .rma import rma_jit, rma_njit, rma_cuda

rsi_name = "rsi"
rsi_result_name = [
    ["rsi", 1],
    ["gains", 0],
    ["losses", 0],
    ["avg_gains", 0],
    ["avg_losses", 0],
]  # 1指标名,2是否显示到dataframe中
rsi_load = [
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
        5,
        # rsi_result, gains_result, losses_result, avg_gains_result, avg_losses_result
    ],  # 表明该指标会输入多少列数据，直接决定了其输入的宽度。
    [
        "indicator_id",
        2,
    ],  # 代表该指标的id,不能跟其他指标重复。
]
rsi_talib = True


def get_rsi_logic(rma_func):
    """
    返回 RSI 计算的核心 Python 逻辑。
    接收已编译的 RMA 函数，填充 RSI 数组。
    """

    def _actual_rsi_logic(
        close,
        period,
        rsi_result,
        gains_result,
        losses_result,
        avg_gains_result,
        avg_losses_result,
    ):
        n = len(close)

        # 初始化 gains 和 losses 数组，第一个元素为 NaN，以匹配 pandas_ta 的 diff 行为
        # 确保全部初始化为 NaN，因为 diff 之后的数据可能不是 0
        gains_result[:] = np.nan
        losses_result[:] = np.nan

        # 计算 gains 和 losses
        for i in range(1, n):
            diff = close[i] - close[i - 1]
            if diff > 0:
                gains_result[i] = diff
                losses_result[i] = 0.0  # 增益时损失为 0
            else:  # diff <= 0
                gains_result[i] = 0.0  # 损失时增益为 0
                losses_result[i] = -diff  # 损失为正值

        # 计算平均 gains 和 losses (使用 RMA)
        # 现在的 rma_func 会在索引 period 处产生第一个有效结果
        rma_func(gains_result, period, avg_gains_result)
        rma_func(losses_result, period, avg_losses_result)

        # 计算 RSI
        for i in range(n):
            # RSI 的第一个有效值应该从索引 `period` 处开始
            if i < period:  # RSI 在前 period 个数据点是 NaN
                rsi_result[i] = np.nan
                continue

            # 确保 avg_gains_result[i] 和 avg_losses_result[i] 不是 NaN
            # Numba 中检查 NaN 的方式：x == x
            if not (
                avg_gains_result[i] == avg_gains_result[i]
                and avg_losses_result[i] == avg_losses_result[i]
            ):
                rsi_result[i] = np.nan
                continue  # 如果平均值是 NaN，则 RSI 也是 NaN

            if avg_losses_result[i] == 0:
                rsi_result[i] = 100.0  # 避免除以零，如果 avg_losses 为 0，RSI 为 100
            else:
                rs = avg_gains_result[i] / avg_losses_result[i]
                rsi_result[i] = 100.0 - (100.0 / (1.0 + rs))

        return (
            rsi_result,
            gains_result,
            losses_result,
            avg_gains_result,
            avg_losses_result,
        )

    return _actual_rsi_logic


# 创建 RSI 函数，复用已编译的 RMA 函数
def create_rsi_function(mode, rma_func_compiled):
    _rsi_py_logic = get_rsi_logic(rma_func_compiled)

    rsi_compiled = numba_factory(
        mode,
        types.Tuple(
            (
                numba_float[:],  # result
                numba_float[:],  # gains_result
                numba_float[:],  # losses_result
                numba_float[:],  # avg_gains_result
                numba_float[:],  # avg_losses_result
            )
        )(
            numba_float[:],  # close
            numba_int,  # period
            numba_float[:],  # result
            numba_float[:],  # gains_result
            numba_float[:],  # losses_result
            numba_float[:],  # avg_gains_result
            numba_float[:],  # avg_losses_result
        ),
    )(_rsi_py_logic)

    return rsi_compiled


# 为不同模式生成核心函数
rsi_jit = create_rsi_function("jit", rma_jit)
rsi_njit = create_rsi_function("njit", rma_njit)
rsi_cuda = create_rsi_function("cuda", rma_cuda)
