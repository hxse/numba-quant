from utils.numba_helpers import numba_factory
from .sma import sma_jit, sma_njit, sma_cuda
from .bbands import (
    bollinger_jit,
    bollinger_njit,
    bollinger_cuda,
)
from .rsi import rsi_jit, rsi_njit, rsi_cuda
from .atr import atr_jit, atr_njit, atr_cuda
from .psar import psar_jit, psar_njit, psar_cuda
from utils.constants import ENABLE_CACHE, np_int, np_float, numba_int, numba_float
import numpy as np


def get_indicator_selector_logic(
    sma_func, bollinger_func, rsi_func, atr_func, psar_func
):
    def _actual_indicator_selector_logic(
        ohlcv_data, child_indicator_params, indicator_load, child_out_arrays
    ):
        time_array = ohlcv_data[:, 0]
        open_array = ohlcv_data[:, 1]
        high_array = ohlcv_data[:, 2]
        low_array = ohlcv_data[:, 3]
        close_array = ohlcv_data[:, 4]

        for i in range(indicator_load.shape[0]):
            # if indicator_load[i, 0]:  # 旧的加载开关已移除
            idx_params = indicator_load[i, 0]
            idx_result = indicator_load[i, 2]
            indicator_id = indicator_load[i, 4]

            if indicator_id == 0:  # SMA
                sma_period = int(child_indicator_params[idx_params + 0])
                sma_result = child_out_arrays[:, idx_result + 0]
                sma_func(close_array, sma_period, sma_result)
            elif indicator_id == 1:  # bbands
                bollinger_period = int(child_indicator_params[idx_params + 0])
                bollinger_std_mult = float(child_indicator_params[idx_params + 1])
                middle_result = child_out_arrays[:, idx_result + 0]
                upper_result = child_out_arrays[:, idx_result + 1]
                lower_result = child_out_arrays[:, idx_result + 2]
                bollinger_func(
                    close_array,
                    bollinger_period,
                    bollinger_std_mult,
                    middle_result,
                    upper_result,
                    lower_result,
                )
            elif indicator_id == 2:  # RSI
                rsi_period = int(child_indicator_params[idx_params + 0])
                rsi_result = child_out_arrays[:, idx_result + 0]
                gains_result = child_out_arrays[:, idx_result + 1]
                losses_result = child_out_arrays[:, idx_result + 2]
                avg_gains_result = child_out_arrays[:, idx_result + 3]
                avg_losses_result = child_out_arrays[:, idx_result + 4]
                rsi_func(
                    close_array,
                    rsi_period,
                    rsi_result,
                    gains_result,
                    losses_result,
                    avg_gains_result,
                    avg_losses_result,
                )
            elif indicator_id == 3:  # ATR
                atr_period = int(child_indicator_params[idx_params + 0])
                atr_result = child_out_arrays[:, idx_result + 0]
                tr_result = child_out_arrays[:, idx_result + 1]
                atr_func(
                    high_array,
                    low_array,
                    close_array,
                    atr_period,
                    atr_result,
                    tr_result,
                )
            elif indicator_id == 4:  # PSAR
                af0 = float(child_indicator_params[idx_params + 0])
                af = float(child_indicator_params[idx_params + 1])
                max_af = float(child_indicator_params[idx_params + 2])
                psar_long_result = child_out_arrays[:, idx_result + 0]
                psar_short_result = child_out_arrays[:, idx_result + 1]
                psar_af_result = child_out_arrays[:, idx_result + 2]
                psar_reversal_result = child_out_arrays[:, idx_result + 3]
                psar_func(
                    high_array,
                    low_array,
                    close_array,
                    af0,
                    af,
                    max_af,
                    psar_long_result,
                    psar_short_result,
                    psar_af_result,
                    psar_reversal_result,
                )

        return child_out_arrays

    return _actual_indicator_selector_logic


def create_indicator_selector_function(
    mode, sma_func, bollinger_func, rsi_func, atr_func, psar_func
):
    _indicator_selector_py_logic = get_indicator_selector_logic(
        sma_func, bollinger_func, rsi_func, atr_func, psar_func
    )

    indicator_selector_compiled = numba_factory(
        mode,
        numba_float[:, :](
            numba_float[:, :],  # ohlcv_data
            numba_float[:],  # child_indicator_params
            numba_int[:, :],  # indicator_load
            numba_float[:, :],  # child_out_arrays
        ),
    )(_indicator_selector_py_logic)
    return indicator_selector_compiled


indicator_selector_jit = create_indicator_selector_function(
    "jit", sma_jit, bollinger_jit, rsi_jit, atr_jit, psar_jit
)
indicator_selector_njit = create_indicator_selector_function(
    "njit", sma_njit, bollinger_njit, rsi_njit, atr_njit, psar_njit
)
indicator_selector_cuda = create_indicator_selector_function(
    "cuda", sma_cuda, bollinger_cuda, rsi_cuda, atr_cuda, psar_cuda
)
