import pandas as pd
import numpy as np
from tqdm import tqdm
from .scMetData import scMetData
from ._utils import extract_mz_intensity_from_scm_events, filter_intensity

# 基于特征密度的聚类，利用特征密度作为聚类的核心标准。
def extract_features_by_hit_rate_guided_clustering(mdata: scMetData, intensity_threshold: float = 500, min_hit_rate: float = 0.1, ppm_threshold: int = 10):
    """
    Hit Rate-Guided Clustering for Metabolic Feature Extraction, HRGC.

    我们引入了命中率(hit rate)的概念用于描述每一个候选特征在所有细胞中出现的频率，命中率越高表明这个候选特征在更多的细胞中出现，可以作为衡量该候选特征视为代谢特征可信程度的指标之一。


    Params:
    -------
    mdata:
    intensity_threshold:
    min_hit_rate:
    ppm_threshold:

    Returns:
    -------
    cell_feature_matrix:
    """
    # Step 1: 筛选高强度的 m/z 数据
    min_hits = round(len(mdata.scm_events) * min_hit_rate)
    data = extract_mz_intensity_from_scm_events(mdata)

    mdata.logger.info(f'Start metabolic features extraction by HRGC')
    mdata.logger.info(f'1) Filter out the m/z data where intensity < {intensity_threshold}.')
    data = filter_intensity(data, intensity_threshold)

    # Step 2: 计算 hits
    mdata.logger.info(f'2) For each potential feature, calculate its hits and hit rate.')
    data = calculate_hits(data, ppm_threshold)

    # Step 3: 基于hits进行聚类。
    mdata.logger.info(f'3) Extract features by hit-rate guided clustering.')
    cell_feature_matrix = cluster_by_hits(data, min_hits=min_hits, ppm_threshold=10)

    mdata.cell_feature_matrix = cell_feature_matrix

    mdata.processing_status['feature_extraction_strategy'] = 'hit rate-guided clustering'
    return mdata


def calculate_hits(data: pd.DataFrame, ppm_threshold: float) -> pd.DataFrame:
    """
    高效计算每个 m/z 的命中数(hits)。
    Args:
        data (pd.DataFrame): 输入数据框，包含 'mz', 'CellNumber' 列。
        ppm_threshold (float): 允许的 m/z 容差范围，单位 ppm。
    Returns:
        pd.DataFrame: 添加了 'hits', 'left_hits', 'right_hits' 列的数据框。
    """
    # 排序数据并初始化
    data = data.sort_values(by='mz').reset_index(drop=True)
    mz_values = data['mz'].values
    cell_numbers = data['CellNumber'].values
    n = len(mz_values)

    # 初始化结果数组
    hits = np.zeros(n, dtype=int)
    left_hits = np.zeros(n, dtype=int)
    right_hits = np.zeros(n, dtype=int)

    # 初始化右边界索引
    right_index = 0

    # 遍历每个 m/z
    for i in tqdm(range(n), desc="Calculating hits"):
        mz = mz_values[i]
        left_bound = mz - mz * ppm_threshold / 1e6
        right_bound = mz + mz * ppm_threshold / 1e6

        # 更新左边界索引
        left_index = i
        while left_index > 0 and mz_values[left_index - 1] >= left_bound:
            left_index -= 1

        # 更新右边界索引
        while right_index < n and mz_values[right_index] <= right_bound:
            right_index += 1

        # 计算支持度，将当前点计入左密度中
        left_cells = set(cell_numbers[left_index:i]) # 包含当前点
        right_cells = set(cell_numbers[i:right_index]) # 不包含当前点
        hits[i] = len(left_cells.union(right_cells))
        left_hits[i] = len(left_cells)
        right_hits[i] = len(right_cells)

    # 添加结果列到 DataFrame
    data['hits'] = hits
    data['left_hits'] = left_hits
    data['right_hits'] = right_hits

    return data

# Step 3: 基于密度和索引进行聚类。
# 根据连续性，初步将聚类划分为不同类型的组别。
def cluster_by_hits(data: pd.DataFrame, min_hits: int=3, ppm_threshold: int=10):
    """
    主函数基于hits进行聚类。

    Params:
    -------
    data: 包含mz,hits 等信息的DataFrame。
    min_hts: 筛选的最小密度值，小于此值的数据直接标记为cluster=0。

    Returns:
    -------
    cell_feature_matrix: 基于特征密度聚类得到的细胞特征矩阵。
    """
    # 初始化cluster列为-1，表示所有点未被聚类。
    data['cluster']  = -1

    # 用于存储聚类相关数据，聚类标记从 1 开始。
    #  初始化clusters
    unique_cells = sorted(data['CellNumber'].unique())
    total_cells = len(unique_cells)

    cell_feature_matrix = pd.DataFrame(columns=[
        'Feature',
        'mz_center',
        'mz_mean',
        'mz_std',
        'mz_median',
        'hits',
        'hit_rate',
    ]+unique_cells)

    cluster_id = 1
    max_hits = data['hits'].max()

    # 筛选低密度点并标记cluster=0
    data.loc[data['hits'] <= 3, 'cluster'] = 0

    # 使用tqdm显示进度
    for hit in tqdm(range(int(max_hits), int(min_hits)-1, -1), desc="基于密度值从小到大进行聚类"):
        # 筛选hits 等于当前密度点且暂未参与聚类的行。
        specific_hit_rows = data[(data['hits'] == hit) & (data['cluster'] == -1)]
        if specific_hit_rows.empty:
            continue
        # 初步连续性聚类
        sub_clusters = perform_continuity_clustering(specific_hit_rows)

        # 合并相邻聚类
        merged_clusters=merge_adjacent_clusters(sub_clusters)

        # 对聚类进行预处理，包括计算聚类中心，并基于聚类中心重新划分聚类范围，最后检查并去除重复的 CellNumber，保留 mz 最近的行。
        processed_clusters = process_clusters(data, merged_clusters, ppm_threshold=ppm_threshold)

        # 更新 cell_feature_matrix，并在 data 中标记聚类状态，同时更新边缘密度。
        cell_feature_matrix, data, cluster_id = update_cell_feature_matrix(processed_clusters, cell_feature_matrix, data, cluster_id=cluster_id, total_cells=total_cells, ppm_threshold=ppm_threshold)

    return cell_feature_matrix


# 按行号连续性对指定密度点集合进行初步聚类。
def perform_continuity_clustering(subset):
    """
    按行号连续性对指定密度点集合进行初步聚类。

    参数:
    - subset: Pandas DataFrame，表示筛选出来的具有相同密度的点集。

    返回:
    - sub_clusters: 一个列表，每个元素是一个子聚类（由行组成的集合）。
    """
    # 计算行号差值，标记新聚类的起点
    subset['group'] = (subset.index.to_series().diff() != 1).cumsum()

    # 根据 group 分组并提取子聚类
    sub_clusters = [group for _, group in subset.groupby('group')]

    # 删除临时列 group
    subset.drop(columns=['group'], inplace=True)

    return sub_clusters

def merge_adjacent_clusters(clusters):
    """
    合并相邻聚类：通过行号差和右密度条件。

    参数:
    - clusters: 一个列表，每个元素是一个聚类（由 DataFrame 行组成的集合）。

    返回:
    - merged_clusters: 合并后的聚类列表。
    """
    merged_clusters = []

    for cluster in clusters:
        if not merged_clusters:
            merged_clusters.append(cluster)
        else:
            last_cluster = merged_clusters[-1]
            last_point = last_cluster.iloc[-1]  # 上一个聚类的最后一个点
            first_point = cluster.iloc[0]  # 当前聚类的第一个点

            # 检查行号差和右密度条件
            if (first_point.name - last_point.name) <= last_point['right_hits']:
                # 合并当前聚类到上一个聚类
                merged_clusters[-1] = pd.concat([last_cluster, cluster], ignore_index=False)
            else:
                # 当前聚类单独存储
                merged_clusters.append(cluster)

    return merged_clusters

# 整理出一个根据基于密度的聚类方法得到的m/z列表出来，包含有一些信息，例如hits。
def process_clusters(data, merged_clusters, ppm_threshold=10):
    """
    对每个聚类执行以下操作：
    1. 计算聚类中心 (mz 中位数)。
    2. 基于聚类中心和 ppm 范围重新划分聚类范围。
    3. 检查并去除重复的 CellNumber，保留 mz 最近的行。

    Params:
    -------
    data:
    merged_clusters: 列表，每个元素是一个DataFrame，表示一个聚类
    ppm_threshold: mz的允许波动范围(ppm)

    Returns:
    --------
    处理后的聚类列表。
    """
    processed_clusters = []

    for cluster in merged_clusters:
        # 1. 计算聚类中心 mz（中位数）
        mz_center = cluster['mz'].median()

        # 2. 根据 10 ppm 范围重新划分聚类范围
        mz_min = mz_center * (1 - ppm_threshold / 1e6)
        mz_max = mz_center * (1 + ppm_threshold / 1e6)
        filtered_data = data[(data['mz'] >= mz_min) & (data['mz'] <= mz_max)]

        # 3. 检查并去除重复的 CellNumber
        filtered_data['mz_center'] = mz_center
        filtered_data['mz_diff'] = abs(filtered_data['mz'] - mz_center)  # 计算与 mz_center 的差值
        filtered_data = (
            filtered_data.sort_values(by=['CellNumber', 'mz_diff'])  # 按 CellNumber 和 mz_diff 排序
            .drop_duplicates(subset='CellNumber', keep='first')      # 删除重复的 CellNumber
            .drop(columns=['mz_diff'])                              # 删除辅助列 mz_diff
        )

        # 保存处理后的聚类
        processed_clusters.append(filtered_data)

    return processed_clusters


def calculate_hits_for_indices(data, indices, ppm_threshold):
    # 提取未聚类的数据
    valid_data = data[data['cluster'] == -1]

    # 保存原始索引
    valid_data['original_index'] = valid_data.index

    # 按mz排序并重新构建索引
    valid_data = valid_data.sort_values(by='mz').reset_index(drop=True)

    # 获取重构后的mz值和cell_numbers
    mz_values = valid_data['mz'].values
    cell_numbers = valid_data['CellNumber'].values

    # 使用原始索引来创建映射
    index_map = {original_idx: new_idx for new_idx, original_idx in enumerate(valid_data['original_index'])}

    # 遍历所有传入的indices
    for original_index in indices:
        # 查找index在重构索引后的位置
        i = index_map[original_index]

        # 获取当前的mz值
        mz = mz_values[i]

        # 计算左边界和右边界
        left_bound = mz - mz * ppm_threshold / 1e6
        right_bound = mz + mz * ppm_threshold / 1e6

        # 初始化左右边界索引
        left_index = i
        right_index = i

        # 更新左边界索引
        while left_index > 0 and mz_values[left_index - 1] >= left_bound:
            left_index -= 1

        # 更新右边界索引
        while right_index < len(mz_values) and mz_values[right_index] <= right_bound:
            right_index += 1

        # 计算左密度和右密度
        # left_cells 包含当前点
        left_cells = set(cell_numbers[left_index:i+1])  # 包含当前点
        # right_cells 不包含当前点
        right_cells = set(cell_numbers[i+1:right_index])  # 不包含当前点

        # 更新data中的相应列
        data.at[original_index, 'left_hits'] = len(left_cells)
        data.at[original_index, 'right_hits'] = len(right_cells)
        data.at[original_index, 'hits'] = len(left_cells.union(right_cells))

    return data


def find_affected_indices(affected_indices, data, min_index, max_index):
    """
    确定受聚类影响需要重新计算密度的点索引。

    Args:
        affected_indices(list): 聚类影响需要重新计算密度的点索引。
        data (pd.DataFrame): 数据表，包含 'left_hits', 'right_hits' 等列。
        min_index (int): 聚类范围的最小索引。
        max_index (int): 聚类范围的最大索引。

    Returns:
        list: 需要重新计算密度的索引列表。
    """
    # 检查左边界扩展
    left_index = min_index - 1
    while left_index >= 0:
        # 检查右密度条件
        if (data.at[left_index, 'right_hits'] == 0 or
            data.at[left_index, 'right_hits'] <= (min_index - left_index) - 1):
            break
        # 只有在 'cluster' 列值为 -1 时才添加该索引
        if data.at[left_index, 'cluster'] == -1:
            affected_indices.append(left_index)
        left_index -= 1

    # 检查右边界扩展
    right_index = max_index + 1
    while right_index < len(data):
        # 检查左密度条件
        if (data.at[right_index, 'left_hits'] == 1 or
            data.at[right_index, 'left_hits'] <= (right_index - max_index)):
            break
        # 只有在 'cluster' 列值为 -1 时才添加该索引
        if data.at[right_index, 'cluster'] == -1:
            affected_indices.append(right_index)
        right_index += 1

    return affected_indices


def update_cell_feature_matrix(processed_clusters, cell_feature_matrix, data, cluster_id, total_cells, ppm_threshold):
    """
    更新 cell_feature_matrix，并在 data 中标记聚类状态，同时更新边缘密度。
    """
    affected_indices = []
    # 将所有 cluster 依次处理
    for cluster in processed_clusters:
        if cluster.empty or 'mz_center' not in cluster.columns:
            continue  # 跳过空的或不完整的 cluster

        mz_center = cluster['mz_center'].iloc[0]
        cellnumber = len(cluster)
        cellratio = cellnumber / total_cells * 100
        mz_mean = cluster['mz'].mean()
        mz_std = cluster['mz'].std()
        mz_median = cluster['mz'].median()

        # 构造单条特征信息
        cluster_info = {
            'Feature': cluster_id,
            'mz_center': mz_center,
            'mz_mean': mz_mean,
            'mz_std':mz_std,
            'mz_median': mz_median,
            'hits': cellnumber,
            'hit_rate': cellratio,
        }
        cluster_info.update({
            row['CellNumber']: row['intensity']
            for _, row in cluster.iterrows()
        })

        # 更新特征矩阵
        cell_feature_matrix = pd.concat([cell_feature_matrix, pd.DataFrame([cluster_info])], ignore_index=True)

        # 更新 data 中的 cluster 标记
        min_index, max_index = cluster.index.min(), cluster.index.max()
        data.loc[min_index:max_index, 'cluster'] = cluster_id

        # 确定受到聚类影响需要重新计算密度的点索引。
        affected_indices = find_affected_indices(affected_indices, data, min_index, max_index)

        # 更新聚类 ID
        cluster_id += 1

    # 根据affected_indices更新data中需要更新的点的密度。返回data.
    unique_indices = set(affected_indices)
    filtered_indices = [idx for idx in unique_indices if data.loc[idx,'cluster'] == -1]

    data = calculate_hits_for_indices(data, filtered_indices, ppm_threshold)
    return cell_feature_matrix, data, cluster_id

