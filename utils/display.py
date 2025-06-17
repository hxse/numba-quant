import numpy as np


def print_array_details(arr, prefix, num=5):
    print(f"{prefix} 数组详情:")
    arr_len = len(arr)

    # 打印前num个元素
    if arr_len <= num:
        print(f"---- 前 {arr_len} 个元素:\n    {arr[:arr_len]}")
    else:
        print(f"---- 前 {num} 个元素:\n    {arr[:num]}")

    # 打印后num个元素
    if arr_len > num:
        print(f"---- 后 {num} 个元素:\n    {arr[-num:]}")

    # 查找第一个非 NaN 元素的索引
    non_nan_indices = np.where(~np.isnan(arr))[0]
    if len(non_nan_indices) > 0:
        first_non_nan_idx = non_nan_indices[0]
        end_idx = min(first_non_nan_idx + num, arr_len)
        print(
            f"---- 从第一个非 NaN 元素 ({first_non_nan_idx} 索引) 开始的 {end_idx - first_non_nan_idx} 个元素:\n    {arr[first_non_nan_idx:end_idx]}"
        )
    else:
        print("---- 数组中没有非 NaN 元素。")
