import numpy as np
from utils.constants import ENABLE_CACHE, np_int, np_float, numba_int, numba_float


def update_indicator_load_cumulative(
    indicator_load: np.ndarray,
    target_col_idx: int,
    source_col_idx: int,
) -> np.ndarray:
    """
    将指定目标列的占位符值（例如 -1），更新为之前所有行的指定源列数值之和所形成的起始索引。

    Args:
        indicator_load (np.ndarray): 一个二维 NumPy 数组，结构为
                                     [[..., 源数据列 (int), ...]]
                                     例如：[[target_idx, source_val, ..., target_idx, source_val]]
                                     此数组应为整数类型，如 np.int32。
        target_col_idx (int): 要填充累积起始索引的目标列的索引 (0-based)。
                              例如，如果要填充第一列，传入 0；如果要填充第三列，传入 2。
        source_col_idx (int): 用于计算累积和的源列的索引 (0-based)。
                              例如，如果要使用第二列的数据计算，传入 1；如果要使用第四列的数据计算，传入 3。

    Returns:
        np.ndarray: 更新后的 indicator_load 数组。
    """
    # 复制数组以避免修改原始输入
    updated_indicator_load = indicator_load.copy()

    # --- 基础输入验证：数组维度和列索引有效性 ---
    if indicator_load.ndim != 2:
        raise ValueError("indicator_load 必须是二维数组。")

    num_cols = updated_indicator_load.shape[1]

    if not (0 <= target_col_idx < num_cols):
        raise ValueError(
            f"target_col_idx ({target_col_idx}) 超出了 indicator_load 的列数范围 [0, {num_cols - 1}]。"
        )
    if not (0 <= source_col_idx < num_cols):
        raise ValueError(
            f"source_col_idx ({source_col_idx}) 超出了 indicator_load 的列数范围 [0, {num_cols - 1}]。"
        )
    # --- 基础输入验证结束 ---

    # --- 新增验证：source_col_idx 所在列不能有小于或等于 0 的值 ---
    # 检查 source_col_idx 列的实际数据
    # 注意：这里检查的是整个 source_col_idx 列
    if np.any(updated_indicator_load[:, source_col_idx] <= 0):
        raise ValueError(
            f"indicator_load 的第 {source_col_idx + 1} 列（索引 {source_col_idx}，即用于累积求和的源列）"
            "不能包含小于或等于 0 的值，因为它们通常代表数量或长度。"
        )
    # --- 新增验证结束 ---

    # 1. 提取需要进行累积和的源列数据 (不再需要 mask)
    values_to_sum = updated_indicator_load[:, source_col_idx]

    # 2. 计算累积和以生成起始索引
    if values_to_sum.size > 0:
        # 在 values_to_sum 前面添加一个0，计算累积和，然后移除最后一个元素，得到起始索引
        indices_to_assign = np.cumsum(np.insert(values_to_sum, 0, 0))[:-1]
    else:
        # 如果 values_to_sum 为空，则没有索引需要赋值，返回空数组
        indices_to_assign = np.array([], dtype=updated_indicator_load.dtype)

    # 3. 将生成的索引序列赋值给目标列
    updated_indicator_load[:, target_col_idx] = indices_to_assign

    return updated_indicator_load


def extract_segmented_params(
    indicator_load: np.ndarray, indicator_params: np.ndarray
) -> np.ndarray:
    """
    根据 indicator_load 的规则，对 indicator_params 的每一行进行分割和选择性提取。
    结果将按 indicator_params 的行进行分组，并将每行中提取的激活片段连接起来。
    最终返回的 NumPy 数组将是一个标准的二维数组，保持与原始 indicator_params 相同的数据类型。

    Args:
        indicator_load (np.ndarray): 二维 NumPy 数组，定义了分割和选择规则。
                                     - 第 0 列 (索引 0): 是否激活 (0/1)。
                                     - 第 4 列 (索引 4): 参数数量 (param_count)，用于分割。
        indicator_params (np.ndarray): 二维 NumPy 数组，待提取的参数数据。

    Returns:
        np.ndarray: 一个标准的二维 NumPy 数组。每一行对应于 indicator_params 中的一行，
                    包含从该行提取并连接起来的所有激活片段。
                    返回数组的 dtype 将与 indicator_params 相同。

    Raises:
        ValueError: 如果输入数组不符合任何验证规则。
        IndexError: 如果内部计算导致尝试提取的范围超出了 indicator_params 的实际边界。
    """
    if indicator_load.ndim != 2 or indicator_params.ndim != 2:
        raise ValueError("indicator_load 和 indicator_params 都必须是二维数组。")

    if indicator_load.shape[1] < 5:
        raise ValueError(
            "indicator_load 至少需要 5 列（索引 0 到 4），才能访问到 param_count（索引 4）。"
        )

    # --- 验证：第五列（param_count）不能有小于 0 的值 ---
    if np.any(indicator_load[:, 4] < 0):
        raise ValueError(
            "indicator_load 的第五列（param_count，索引 4）不能包含小于 0 的值。"
        )
    # --- 验证结束 ---

    # 获取 indicator_load 中 param_count (索引 4) 的值
    param_counts_from_load = indicator_load[:, 4].astype(indicator_load.dtype)

    # 验证：indicator_params 的列数必须等于 indicator_load 第5列的累积和
    num_rows_params, num_cols_params = indicator_params.shape
    total_param_sum_from_load = np.sum(param_counts_from_load)

    if num_cols_params != total_param_sum_from_load:
        raise ValueError(
            f"验证失败：indicator_params 的列数 ({num_cols_params}) "
            f"不等于 indicator_load 第五列 (索引 4) 的累积和 ({total_param_sum_from_load})。"
            f"每行 indicator_params 都必须能被这些 param_count 值完整分割。"
        )

    # 计算每个段的起始位置（在每行内部的列偏移量）
    segment_start_offsets = np.cumsum(np.insert(param_counts_from_load[:-1], 0, 0))

    # 获取激活状态
    is_active_rules = indicator_load[:, 0] == 1  # 索引 0: 是否激活

    # 过滤出激活的规则的 param_count 和起始偏移量
    active_param_counts = param_counts_from_load[is_active_rules]
    active_start_offsets = segment_start_offsets[is_active_rules]

    # 计算每行最终提取到的总列数
    num_extracted_cols_per_row = np.sum(active_param_counts)

    # 存储每行处理后的结果
    connected_segments_per_row = []

    # 遍历 indicator_params 的每一行
    for param_row_idx in range(num_rows_params):
        current_param_row = indicator_params[param_row_idx]

        # 存储当前行中所有激活并提取的片段
        current_row_segments = []

        # 对于当前行，基于激活规则和计算出的偏移量进行提取
        for i in range(active_param_counts.size):
            param_count = active_param_counts[i]
            start_offset = active_start_offsets[i]

            if param_count > 0:  # 确保提取数量大于0
                extracted_segment = current_param_row[
                    start_offset : start_offset + param_count
                ]
                current_row_segments.append(extracted_segment)

        # 将当前行提取到的所有片段连接起来
        # 如果 current_row_segments 为空（即 num_extracted_cols_per_row 为 0），
        # np.concatenate([]) 会抛出 ValueError。
        # 此时应该生成一个正确类型的空数组。
        if num_extracted_cols_per_row == 0:
            connected_segment = np.array([], dtype=indicator_params.dtype)
        else:
            # 如果有提取到数据，并且确保了 total_param_sum_from_load 匹配，
            # 那么这里的 concatenate 一定不会出错。
            connected_segment = np.concatenate(current_row_segments)

        connected_segments_per_row.append(connected_segment)

    # 直接将列表中的 NumPy 数组堆叠成一个二维数组
    # 由于所有元素的长度一致（由验证和 num_extracted_cols_per_row 保证），
    # np.array() 会自动处理成正确的二维形状。
    # 如果 connected_segments_per_row 为空（比如 num_rows_params=0），
    # 需要处理以确保返回 (0, N) 或 (0, 0)
    if num_rows_params == 0:  # 如果 indicator_params 本身就没有行
        return np.array([], dtype=indicator_params.dtype).reshape(
            0, num_extracted_cols_per_row
        )

    return np.array(connected_segments_per_row, dtype=indicator_params.dtype)


def extract_segmented_array_by_load_params(
    input_array: np.ndarray, np_load: np.ndarray
) -> list:
    """
    根据 np_load 数组的第二列（索引 1）来从一维数组中提取分段数据。

    Args:
        input_array (np.ndarray): 待提取数据的一维 NumPy 数组。
                                  例如：array([14., 20., 14.,  2., 14.])
        np_load (np.ndarray): 二维 NumPy 数组，其第二列（索引 1）定义了每个段的长度。
                              例如：array([[0, 1, 0, 1, 0],
                                           [1, 1, 1, 1, 0],
                                           [2, 2, 2, 3, 1],
                                           [4, 1, 5, 2, 3]])

    Returns:
        list: 包含分段数据的嵌套列表。
              例如：[[14], [20], [14, 2], [14]]
    """
    if input_array.ndim != 1:
        raise ValueError("input_array 必须是一维数组。")
    if np_load.ndim != 2:
        raise ValueError("np_load 必须是二维数组。")
    if np_load.shape[1] < 2:
        raise ValueError("np_load 至少需要 2 列才能访问索引 1。")

    segmented_data = []
    current_offset = 0
    for row in np_load:
        segment_length = row[1]  # 获取第二列的值作为段的长度
        if segment_length > 0:
            # 确保不会超出 input_array 的边界
            if current_offset + segment_length > len(input_array):
                raise IndexError(
                    f"尝试提取的段超出了 input_array 的边界。 "
                    f"当前偏移量: {current_offset}, 段长度: {segment_length}, "
                    f"input_array 长度: {len(input_array)}"
                )
            segment = input_array[
                current_offset : current_offset + segment_length
            ].tolist()
            segmented_data.append(segment)
            current_offset += segment_length
        else:
            segmented_data.append([])  # 如果长度为0，添加空列表
    return segmented_data


def extract_segmented_2d_array_by_load_params(
    input_2d_array: np.ndarray, np_load: np.ndarray
) -> list:
    """
    根据 np_load 数组的第二列（索引 1）来从二维数组的每一行中提取分段数据。
    此函数会遍历 input_2d_array 的每一行，并对每行应用 extract_segmented_array_by_load_params。

    Args:
        input_2d_array (np.ndarray): 待提取数据的二维 NumPy 数组。
                                     例如：array([[14., 20., 14.,  2., 14.],
                                                  [20., 50., 20.,  2., 20.]])
        np_load (np.ndarray): 二维 NumPy 数组，其第二列（索引 1）定义了每个段的长度。
                              例如：array([[0, 1, 0, 1, 0],
                                           [1, 1, 1, 1, 0],
                                           [2, 2, 2, 3, 1],
                                           [4, 1, 5, 2, 3]])

    Returns:
        list: 包含分段数据的嵌套列表的列表。
              例如：[[[14], [20], [14, 2], [14]], [[20], [50], [20, 2], [20]]]
    """
    if input_2d_array.ndim != 2:
        raise ValueError("input_2d_array 必须是二维数组。")
    if np_load.ndim != 2:
        raise ValueError("np_load 必须是二维数组。")
    if np_load.shape[1] < 2:
        raise ValueError("np_load 至少需要 2 列才能访问索引 1。")

    all_segmented_data = []
    for row_idx in range(input_2d_array.shape[0]):
        # 对二维数组的每一行调用一维提取函数
        segmented_row_data = extract_segmented_array_by_load_params(
            input_2d_array[row_idx], np_load
        )
        all_segmented_data.append(segmented_row_data)
    return all_segmented_data
