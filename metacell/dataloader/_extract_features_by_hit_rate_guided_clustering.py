import pandas as pd
import numpy as np
from tqdm import tqdm
from .scMetData import scMetData
from ._utils import extract_mz_intensity_from_scm_events, filter_intensity
from ._annotate_metabolite_feature import annotate_metabolites_feature
# 基于特征密度的聚类，利用特征密度作为聚类的核心标准。
def extract_features_by_hit_rate_guided_clustering(mdata: scMetData, intensity_threshold: float = 200, min_hit_rate: float = 0.1, offset: int = 1,ppm_threshold: int = 10):
    """
    Hit Rate-Guided Clustering for Metabolic Feature Extraction, HRGC.

    我们引入了命中率(hit rate)的概念用于描述每一个候选特征在所有细胞中出现的频率，命中率越高表明这个候选特征在更多的细胞中出现，可以作为衡量该候选特征视为代谢特征可信程度的指标之一。
    Params:
    -------
    mdata:
    intensity_threshold:
    min_hit_rate:
    offset:
    ppm_threshold:

    Returns:
    -------
    cell_feature_matrix:
    """
    # Step 1: 筛选高强度的 m/z 数据
    min_hits = round(len(mdata.scm_events) * min_hit_rate)
    data = extract_mz_intensity_from_scm_events(mdata, offset=offset, ppm_threshold=ppm_threshold)

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
    mdata.cell_feature_matrix.insert(8, 'HHI', mdata.cell_feature_matrix.pop('HHI'))
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
    mz_values = data['mz'].values.reshape(-1, 1)
    cell_numbers = data['CellNumber'].values
    n = len(mz_values)

    # 使用KDTree加速邻域查询
    from scipy.spatial import cKDTree
    tree = cKDTree(mz_values)
    ppm_ranges = mz_values * ppm_threshold / 1e6

    hits = np.zeros(n, dtype=int)
    left_hits = np.zeros(n, dtype=int)
    right_hits = np.zeros(n, dtype=int)

    # 查询每个点的邻域
    for i in range(n):
        mz = mz_values[i][0]
        left_bound = mz - ppm_ranges[i][0]
        right_bound = mz + ppm_ranges[i][0]
        # 查询所有在ppm范围内的索引
        idx = tree.query_ball_point(mz, r=ppm_ranges[i][0])
        # 统计命中cell
        unique_cells = set(cell_numbers[idx])
        hits[i] = len(unique_cells)
        # 左右密度
        left_idx = [j for j in idx if mz_values[j][0] <= mz]
        right_idx = [j for j in idx if mz_values[j][0] > mz]
        left_hits[i] = len(set(cell_numbers[left_idx]))
        right_hits[i] = len(set(cell_numbers[right_idx]))

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
    # 提前排序，减少后续重复排序
    data = data.sort_values(by='mz').reset_index(drop=True)
    data['cluster']  = -1

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
        'HHI',
    ]+unique_cells)

    cluster_id = 1
    max_hits = data['hits'].max()

    # 筛选低密度点并标记cluster=0
    data.loc[data['hits'] <= 3, 'cluster'] = 0
    all_hhi_values = []
    for hit in tqdm(range(int(max_hits), int(min_hits)-1, -1), desc="Clustering by hit rate (high to low)"):
        specific_hit_rows = data[(data['hits'] == hit) & (data['cluster'] == -1)]
        if specific_hit_rows.empty:
            continue
        sub_clusters = perform_continuity_clustering(specific_hit_rows)
        merged_clusters = merge_adjacent_clusters(sub_clusters)
        processed_clusters = process_clusters_optimized(data, merged_clusters, ppm_threshold=ppm_threshold)
        cell_feature_matrix, data, cluster_id = update_cell_feature_matrix(processed_clusters, cell_feature_matrix, data, cluster_id=cluster_id, total_cells=total_cells, ppm_threshold=ppm_threshold)
        for cluster in processed_clusters:
            hhi_value = calculate_HHI(cluster, hit)
            all_hhi_values.append(hhi_value)
    cell_feature_matrix['HHI'] = all_hhi_values
    cell_feature_matrix.insert(8, 'HHI', cell_feature_matrix.pop('HHI'))
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

# 优化后的process_clusters，批量赋值避免SettingWithCopyWarning
def process_clusters_optimized(data, merged_clusters, ppm_threshold=10):
    processed_clusters = []
    for cluster in merged_clusters:
        mz_center = cluster['mz'].median()
        mz_min = mz_center * (1 - ppm_threshold / 1e6)
        mz_max = mz_center * (1 + ppm_threshold / 1e6)
        filtered_idx = data.index[(data['mz'] >= mz_min) & (data['mz'] <= mz_max) & ((data['cluster'] == -1))]
        filtered_data = data.loc[filtered_idx].copy()
        if not filtered_data.empty:
            filtered_data.loc[:, 'mz_center'] = mz_center
            filtered_data.loc[:, 'mz_diff'] = abs(filtered_data['mz'] - mz_center)
            filtered_data = (
                filtered_data.sort_values(by=['CellNumber', 'mz_diff'])
                .drop_duplicates(subset='CellNumber', keep='first')
                .drop(columns=['mz_diff'])
            )
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



def calculate_HHI(cluster, hit):
    """
    Hit Homogeneity Index (HHI)
    Meaning: Measures the homogeneity or consistency of hits within the cluster. A value closer to 1 indicates that the hits values between samples are more consistent.
    """
    # Calculate the relative difference for each row (1 - hits / hit)
    cluster['relative_diff'] = 1 - (cluster['hits'] / hit)

    # Calculate the square of the relative difference
    cluster['relative_diff_squared'] = cluster['relative_diff'] ** 2

    # Calculate the mean of the squared differences and take the square root to get the standard deviation
    std_diff = np.sqrt(cluster['relative_diff_squared'].mean())

    return 1 - std_diff

def remove_and_update_matrix(mdata, removed_list, metab_anno, intensity_threshold=200, min_hit_rate=0.1, offset=1, ppm_threshold=10):
    """
    根据 removed_list 更新 mdata.scm_events_index 和 mdata.scm_events，
    并重新运行特征提取和代谢物注释。
    Parameters
    ----------
    mdata : object
        包含 scm_events_index (dict) 和 scm_events (DataFrame) 的对象
    removed_list : list or array-like
        1-based 索引列表（需要先转 0-based 再删除）
    intensity_threshold, min_hit_rate, offset, ppm_threshold : 参数
        用于 mc.dl.extract_features_by_hit_rate_guided_clustering
    metab_anno : object
        用于 mc.dl.annotate_metabolites_feature 的注释信息

    Returns
    -------
    mdata : object
        更新后的 mdata
    """
    removed_list = np.array(removed_list) - 1
    
    for key in mdata.scm_events_index:
        mdata.scm_events_index[key] = np.delete(mdata.scm_events_index[key], removed_list)
    
    mdata.scm_events = mdata.scm_events.drop(
        mdata.scm_events.index[removed_list]
    ).reset_index(drop=True)

    if "CellNumber" in mdata.scm_events.columns:
        n = len(mdata.scm_events)
        mdata.scm_events["CellNumber"] = [f"Cell{str(i).zfill(5)}" for i in range(1, n + 1)]

    mdata = extract_features_by_hit_rate_guided_clustering(
        mdata,
        intensity_threshold=intensity_threshold,
        min_hit_rate=min_hit_rate,
        offset=offset,
        ppm_threshold=ppm_threshold
    )
    mdata = annotate_metabolites_feature(mdata, metab_anno)
    return mdata