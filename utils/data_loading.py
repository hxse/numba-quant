import numpy as np
import pandas as pd
from utils.constants import ENABLE_CACHE, np_int, np_float, numba_int, numba_float

tohlcv_name = ["time", "open", "high", "low", "close", "volume"]


def load_tohlcv_from_csv(file_path: str, data_size: int = None) -> np.ndarray:
    """
    从 CSV 文件加载 OHLCV 数据并进行处理。

    Args:
        file_path (str): CSV 文件的路径。
        data_size (int): 需要加载的数据大小。

    Returns:
        np.ndarray: 处理后的 OHLCV 数据。
    """
    df = pd.read_csv(file_path)
    # 将 timestamp 列重命名为 time
    if "timestamp" in df.columns:
        df.rename(columns={"timestamp": "time"}, inplace=True)
    # 将数值列转换为 np_float
    numeric_cols = tohlcv_name
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(np_float)
    df["date"] = pd.to_datetime(df["time"], unit="ms")

    if data_size and len(df) > data_size:
        df = df.iloc[:data_size]

    return df


def convert_tohlcv_numpy(df):
    return df[tohlcv_name].to_numpy().astype(np_float)


def convert_to_pandas_dataframe(
    segmented_params: list,
    result_name: list,
    child_out_array: np.ndarray,
    filter_columns: bool = True,
) -> pd.DataFrame:
    """
    将 Numba 计算结果数组转换为 Pandas DataFrame。

    Args:
        segmented_params (list): 嵌套列表，包含每个指标组的参数。
                                 例如：[[14], [20], [14, 2], [14]]
        result_name (list): 嵌套列表，包含每个指标组的原始列名。
                                   例如：[['sma', 1], ['sma', 1], ['bbm', 1], ['bbu', 1], ['bbl', 1]], ['atr', 1], ['tr', 0]]
                                   第二个元素（0 或 1）表示该列是否应显示在 Pandas DataFrame 中。
        child_out_array (np.ndarray): 单个参数组合的计算结果数组，形状为 (时间点数量, 指标字段数量)。
        filter_columns (bool): 是否根据 result_name 中的显示标志过滤列。默认为 True。

    Returns:
        pd.DataFrame: 包含指标结果的 Pandas DataFrame。

    Raises:
        ValueError: 如果列名数量与数组的第三个维度不匹配。
    """
    final_column_names = []
    columns_to_include_indices = []
    current_col_idx = 0

    for i, sublist_names_with_flags in enumerate(result_name):
        params_for_indicator = segmented_params[i]
        param_suffix = ""
        if params_for_indicator:
            param_suffix = "_" + "_".join(
                map(str, [
                    int(p) if p == int(p) else p for p in params_for_indicator
                ]))

        for name, display_flag in sublist_names_with_flags:
            if filter_columns and display_flag == 0:
                # 如果启用过滤且 display_flag 为 0，则跳过此列
                pass
            else:
                final_column_names.append(f"{name}{param_suffix}")
                columns_to_include_indices.append(current_col_idx)
            current_col_idx += 1

    # 验证要包含的列的数量是否与实际提取的列数匹配
    if len(columns_to_include_indices) != len(final_column_names):
        raise ValueError("内部错误：要包含的列索引数量与最终列名数量不匹配。")

    # 确保 child_out_array 至少有足够的列
    if child_out_array.shape[1] < current_col_idx:
        raise ValueError(
            f"结果数组的列数 ({child_out_array.shape[1]}) 小于预期的总列数 ({current_col_idx})。"
        )

    # 提取需要包含的列
    filtered_out_array = child_out_array[:, columns_to_include_indices]

    # 验证最终列名数量与过滤后的 out_array 的列数是否匹配
    if len(final_column_names) != filtered_out_array.shape[1]:
        raise ValueError(
            f"生成的列名数量 ({len(final_column_names)}) 与过滤后的结果数组的列数 ({filtered_out_array.shape[1]}) 不匹配。"
        )

    # 创建 DataFrame
    df = pd.DataFrame(filtered_out_array, columns=final_column_names)
    return df
