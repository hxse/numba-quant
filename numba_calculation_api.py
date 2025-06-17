import numpy as np
from numba import cuda
from utils.numba_helpers import timer
from utils.indicator_processing import (
    update_indicator_load_cumulative,
    extract_segmented_params,
)
from indicators.indicators_config import create_params

from numba_parallel_kernels import (
    cpu_parallel_calc_jit,
    cpu_parallel_calc_njit,
    gpu_kernel_device,
)
from utils.data_loading import load_ohlcv_from_csv, convert_ohlcv_numpy
from utils.constants import ENABLE_CACHE, np_int, np_float, numba_int, numba_float


def calculate_jit(ohlcv_data, indicator_params, indicator_load):
    """JIT 计算的核心逻辑，返回布林带结果"""

    # 计算指标总列数
    total_indicator_cols = int(np.sum(indicator_load[:, 3]))

    # out_arrays 定义结果数组的形状,例如(2, 200000, 4),分别代表2种参数组合,20万个时间点数据,4个指标字段
    out_shape = (indicator_params.shape[0], ohlcv_data.shape[0], total_indicator_cols)

    # 创建NaN数组
    out_arrays = np.full(out_shape, np.nan, dtype=np_float)

    cpu_parallel_calc_jit(ohlcv_data, indicator_params, indicator_load, out_arrays)
    return out_arrays


@timer
def calculate_jit_timer(ohlcv_data, indicator_params, indicator_load):
    return calculate_jit(ohlcv_data, indicator_params, indicator_load)


def calculate_njit(ohlcv_data, indicator_params, indicator_load):
    """NJIT 计算的核心逻辑，返回布林带结果"""
    # 计算指标总列数
    total_indicator_cols = int(np.sum(indicator_load[:, 3]))

    # 定义结果数组的形状
    out_shape = (indicator_params.shape[0], ohlcv_data.shape[0], total_indicator_cols)

    # 创建NaN数组
    out_arrays = np.full(out_shape, np.nan, dtype=np_float)

    cpu_parallel_calc_njit(ohlcv_data, indicator_params, indicator_load, out_arrays)
    return out_arrays


@timer
def calculate_njit_timer(ohlcv_data, indicator_params, indicator_load):
    return calculate_njit(ohlcv_data, indicator_params, indicator_load)


def calculate_cuda(ohlcv_data, indicator_params, indicator_load):
    """CUDA 计算的核心逻辑，返回布林带结果"""
    # 计算指标总列数
    total_indicator_cols = int(np.sum(indicator_load[:, 3]))

    # 定义结果数组的形状
    out_shape = (indicator_params.shape[0], ohlcv_data.shape[0], total_indicator_cols)

    gpu_data = cuda.to_device(ohlcv_data)
    gpu_indicator_params = cuda.to_device(indicator_params)
    gpu_indicator_load = cuda.to_device(indicator_load)
    host_out_arrays = np.full(out_shape, np.nan, dtype=np_float)
    gpu_out_arrays = cuda.to_device(host_out_arrays)

    threadsperblock = 128
    blockspergrid = (
        indicator_params.shape[0] + (threadsperblock - 1)
    ) // threadsperblock
    if blockspergrid == 0:
        blockspergrid = 1

    gpu_kernel_device[blockspergrid, threadsperblock](
        gpu_data, gpu_indicator_params, gpu_indicator_load, gpu_out_arrays
    )
    cuda.synchronize()

    return gpu_out_arrays.copy_to_host()


@timer
def calculate_cuda_timer(ohlcv_data, indicator_params, indicator_load):
    return calculate_cuda(ohlcv_data, indicator_params, indicator_load)


@timer
def precompiled(ohlcv_data_full, config, data_size=10):
    np_params, np_load, result_name, segmented_params = create_params(config)

    print("预编译 Numba 函数...")

    # 自动识别 ohlcv_data_full 的内存布局并适配
    if ohlcv_data_full.flags["C_CONTIGUOUS"]:
        ohlcv_data = np.ascontiguousarray(ohlcv_data_full[:data_size]).astype(np_float)
        print("预编译数据转换为 C-contiguous 布局。")
    elif ohlcv_data_full.flags["F_CONTIGUOUS"]:
        ohlcv_data = np.asfortranarray(ohlcv_data_full[:data_size]).astype(np_float)
        print("预编译数据转换为 F-contiguous 布局。")
    else:
        # 如果既不是C也不是F，则默认转换为C-contiguous
        ohlcv_data = np.ascontiguousarray(ohlcv_data_full[:data_size]).astype(np_float)
        print("预编译数据转换为 C-contiguous 布局 (默认)。")

    calculate_jit(ohlcv_data, np_params, np_load)
    calculate_njit(ohlcv_data, np_params, np_load)
    calculate_cuda(ohlcv_data, np_params, np_load)
    print("预编译完成。")
