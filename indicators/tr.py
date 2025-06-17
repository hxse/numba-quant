import numpy as np
from numba import types
from utils.numba_helpers import numba_factory
from utils.constants import numba_float


def get_tr_logic():
    """
    返回真实波动范围 (True Range, TR) 的核心 Python 逻辑。
    计算整个数据数组的 TR，填充到结果数组。
    """

    def _actual_tr_logic(high, low, close, tr_result):
        n = len(high)
        # 对于第一个数据点，prev_close 不存在，TR 应该为 np.nan
        if n > 0:
            tr_result[0] = np.nan  # 保持不变

        for i in range(1, n):
            # 检查输入值是否为 np.nan，如果任一为 NaN，则结果为 NaN
            # 在 Numba 中，对于浮点数 NaN 的检查，更常用的是 x != x
            if high[i] != high[i] or low[i] != low[i] or close[i - 1] != close[i - 1]:
                tr_result[i] = np.nan
                continue

            # 计算 TR
            range1 = high[i] - low[i]
            range2 = abs(high[i] - close[i - 1])
            range3 = abs(low[i] - close[i - 1])
            tr_result[i] = max(range1, range2, range3)
        return tr_result

    return _actual_tr_logic


# 创建 TR 函数
def create_tr_function(mode):
    _tr_py_logic = get_tr_logic()
    tr_func_compiled = numba_factory(
        mode,
        numba_float[:](numba_float[:], numba_float[:], numba_float[:], numba_float[:]),
    )(_tr_py_logic)
    return tr_func_compiled


# 为不同模式生成核心函数
tr_jit = create_tr_function("jit")
tr_njit = create_tr_function("njit")
tr_cuda = create_tr_function("cuda")
