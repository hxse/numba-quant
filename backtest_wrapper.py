from numba import void
from utils.numba_helpers import numba_factory
from utils.constants import np_float, numba_float, numba_int
from indicators.indicator_selector import (
    indicator_selector_jit,
    indicator_selector_njit,
    indicator_selector_cuda,
)


def get_backtest_selector_logic(selector_func):
    def _actual_backtest_selector_logic(
        ohlcv_data, child_indicator_params, indicator_load, child_out_arrays
    ):
        selector_func(
            ohlcv_data, child_indicator_params, indicator_load, child_out_arrays
        )

    return _actual_backtest_selector_logic


def create_backtest_selector_function(mode, selector_func):
    _backtest_selector_py_logic = get_backtest_selector_logic(selector_func)

    backtest_selector_compiled = numba_factory(
        mode,
        void(
            numba_float[:, :],  # ohlcv_data
            numba_float[:],  # child_indicator_params
            numba_int[:, :],  # indicator_load
            numba_float[:, :],  # child_out_arrays
        ),
    )(_backtest_selector_py_logic)
    return backtest_selector_compiled


wrapper_indicator_selector_jit = create_backtest_selector_function(
    "jit", indicator_selector_jit
)
wrapper_indicator_selector_njit = create_backtest_selector_function(
    "njit", indicator_selector_njit
)
wrapper_indicator_selector_cuda = create_backtest_selector_function(
    "cuda", indicator_selector_cuda
)
