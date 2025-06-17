from numba import jit, njit, cuda, prange
from utils.constants import ENABLE_CACHE
from backtest_wrapper import (
    wrapper_indicator_selector_jit,
    wrapper_indicator_selector_njit,
    wrapper_indicator_selector_cuda,
)


# 修改并发函数以支持参数组合
@jit(parallel=True, nopython=True, cache=ENABLE_CACHE)
def cpu_parallel_calc_jit(ohlcv_data, indicator_params, indicator_load, out_arrays):
    for idx in prange(indicator_params.shape[0]):
        wrapper_indicator_selector_jit(
            ohlcv_data, indicator_params[idx], indicator_load, out_arrays[idx]
        )


@njit(parallel=True, cache=ENABLE_CACHE)
def cpu_parallel_calc_njit(ohlcv_data, indicator_params, indicator_load, out_arrays):
    for idx in prange(indicator_params.shape[0]):
        wrapper_indicator_selector_njit(
            ohlcv_data, indicator_params[idx], indicator_load, out_arrays[idx]
        )


@cuda.jit
def gpu_kernel_device(ohlcv_data, indicator_params, indicator_load, out_arrays):
    idx = cuda.grid(1)
    if idx < indicator_params.shape[0]:
        wrapper_indicator_selector_cuda(
            ohlcv_data, indicator_params[idx], indicator_load, out_arrays[idx]
        )
