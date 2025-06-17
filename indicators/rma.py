import numpy as np
from numba import types
from utils.numba_helpers import numba_factory
from utils.constants import ENABLE_CACHE, np_int, np_float, numba_int, numba_float


def get_rma_logic():
    """
    返回 RMA 计算的核心 Python 逻辑。
    接收已编译的 SMA 函数，填充中轨 (SMA)、上轨和下轨数组。
    """

    def _actual_rma_logic(close, period, rma_result):
        n = len(close)

        # 填充前 period 个结果为 NaN。
        # 对应 gains/losses 数组，close[0] 是 NaN。
        # pandas_ta 的 RSI 内部 RMA 通常在索引 `period` 处产生第一个有效值。
        for i in range(
            min(period + 1, n)
        ):  # 填充 0 到 period 的 NaN, 这样第一个有效值在 period+1
            rma_result[i] = np.nan

        # 计算第一个有效的 RMA 值
        # 对于 RSI 的 gains/losses，close[0] 是 NaN。
        # 所以第一个有效 RMA 值应基于 close[1] 到 close[period] 这 period 个数据。
        # 结果应放置在索引 `period` 处。
        if n >= period + 1:  # 确保有足够的数据来计算第一个SMA (从索引1到period)
            initial_sum = 0.0
            # 循环从索引 1 开始，到索引 period (包含)，总共 period 个元素
            for i in range(1, period + 1):
                # Numba 中检查 NaN 的方式：x == x
                if close[i] == close[i]:
                    initial_sum += close[i]
                else:
                    # 如果这 period 个元素中包含了 NaN，那么第一个 RMA 也是 NaN
                    initial_sum = np.nan
                    break

            if initial_sum == initial_sum:  # 再次检查是否为 NaN
                rma_result[period] = initial_sum / period
            else:
                rma_result[period] = np.nan  # 如果 initial_sum 变成 NaN，则结果为 NaN

            # 从 period + 1 索引开始计算后续 RMA 值
            for i in range(period + 1, n):
                # 检查前一个 RMA 值是否为 NaN
                if rma_result[i - 1] == rma_result[i - 1]:
                    # 检查当前 close[i] 是否为 NaN
                    if close[i] == close[i]:
                        rma_result[i] = (
                            rma_result[i - 1] * (period - 1) + close[i]
                        ) / period
                    else:
                        # 如果当前值是 NaN，则结果也变成 NaN (NaN 传播)
                        rma_result[i] = np.nan
                else:
                    # 如果前一个结果是 NaN，则当前结果也是 NaN
                    rma_result[i] = np.nan
        else:
            # 如果数据不足以计算第一个有效 RMA (n < period + 1)，则全部为 NaN
            for i in range(n):
                rma_result[i] = np.nan

        return rma_result

    return _actual_rma_logic


# 创建 RMA 函数
def create_rma_function(mode):
    _rma_py_logic = get_rma_logic()

    rma_compiled = numba_factory(
        mode,
        numba_float[:](
            numba_float[:],
            numba_int,
            numba_float[:],
        ),
    )(_rma_py_logic)

    return rma_compiled


# 为不同模式生成核心函数
rma_jit = create_rma_function("jit")
rma_njit = create_rma_function("njit")
rma_cuda = create_rma_function("cuda")
