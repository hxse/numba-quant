import numpy as np
import pandas as pd
import pandas_ta as ta
from numba import cuda
from utils.display import print_array_details
from utils.indicator_processing import (
    update_indicator_load_cumulative,
    extract_segmented_params,
)
import time

from indicators.indicators_config import create_params, config

from utils.data_loading import (
    load_ohlcv_from_csv,
    convert_ohlcv_numpy,
    convert_to_pandas_dataframe,
)
from utils.constants import (
    ENABLE_CACHE,
    ENABLE_precompiled,
    np_int,
    np_float,
    numba_int,
    numba_float,
)


start_time = time.time()

from numba_calculation_api import (
    calculate_jit_timer,
    calculate_njit_timer,
    calculate_cuda_timer,
    precompiled,
)


end_time = time.time()
print(f"numba模块导入冷启动时间: {end_time - start_time:.4f} 秒")


if __name__ == "__main__":
    data_size = 1000 * 20

    csv_file_path = "csv/BTC_USDT/future live/4h/2022-01-01 00_00_00.csv"

    df = load_ohlcv_from_csv(csv_file_path, data_size)

    ohlcv_numpy = convert_ohlcv_numpy(df)

    np_params, np_load, result_name, segmented_params = create_params(config)

    print("#### CUDA 可用性检测 ####")
    if cuda.is_available():
        print("CUDA 可用")
    else:
        print("CUDA 不可用")
        exit()

    if ENABLE_CACHE:
        print("启用缓存")
    if ENABLE_precompiled:
        print("启用预编译")
        precompiled(ohlcv_numpy, config)

    print("\n#### 开始运行 ####\n")

    print("#### 运行 CPU (@jit 并行) ####\n")
    # out_arrays 定义结果数组的形状,例如(2, 200000, 4),分别代表2种参数组合,20万个时间点数据,4个指标字段
    start_time = time.time()
    out_arrays = calculate_jit_timer(ohlcv_numpy, np_params, np_load)
    end_time = time.time()

    print(f"calculate_jit_timer 冷启动时间: {end_time - start_time:.4f} 秒")
    print(f"out_arrays shape: {out_arrays.shape}")
    for select in range(np_params.shape[0]):
        # 将结果转换为 Pandas DataFrame
        df_result = convert_to_pandas_dataframe(
            segmented_params[select], result_name, out_arrays[select]
        )
        print("-" * 30)
        print(f"@jit 参数组合{select} 的结果 DataFrame:")
        print(df_result)
        import pdb

        pdb.set_trace()

    print("#### 运行 CPU (@njit 并行) ####\n")
    start_time = time.time()
    out_arrays = calculate_njit_timer(ohlcv_numpy, np_params, np_load)
    end_time = time.time()
    print(f"calculate_njit_timer 冷启动时间: {end_time - start_time:.4f} 秒")
    print(f"out_arrays shape: {out_arrays.shape}")
    for select in range(np_params.shape[0]):
        # 将结果转换为 Pandas DataFrame
        df_result = convert_to_pandas_dataframe(
            segmented_params[select], result_name, out_arrays[select]
        )
        print("-" * 30)
        print(f"@njit 参数组合{select} 的结果 DataFrame:")
        print(df_result)

    print("#### 运行 GPU (调用 @cuda.jit(device=True) 函数) ####\n")
    start_time = time.time()
    out_arrays = calculate_cuda_timer(ohlcv_numpy, np_params, np_load)
    end_time = time.time()
    print(f"calculate_cuda_timer 冷启动时间: {end_time - start_time:.4f} 秒")
    print(f"out_arrays shape: {out_arrays.shape}")
    for select in range(np_params.shape[0]):
        # 将结果转换为 Pandas DataFrame
        df_result = convert_to_pandas_dataframe(
            segmented_params[select], result_name, out_arrays[select]
        )
        print("-" * 30)
        print(f"@cuda.jit 参数组合{select} 的结果 DataFrame:")
        print(df_result)
