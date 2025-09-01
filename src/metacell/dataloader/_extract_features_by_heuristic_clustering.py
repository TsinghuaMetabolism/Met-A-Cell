import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import gaussian_kde
from .scMetData import scMetData
from ._utils import extract_mz_intensity_from_scm_events, filter_intensity, mz_threshold


# 启发式排序扫描聚类，通过启发式规则对排序后的数据进行逐步聚类。
def extract_features_by_heuristic_clustering(mdata:scMetData, intensity_threshold: float=500, ppm_threshold:float=10, min_cluster_size: int =3):
    """
        启发式排序扫描聚类，通过启发式规则对排序后的数据进行逐步聚类。
        1) 首先对所有细胞的m/z进行从小到大排序，从最小的m/z开始，将其纳入为第一个特征聚类，计算聚类中的m/z的平均值作为聚类中心以及聚类边界。
        2) 往后扫描，对于下一个m/z判断其是否仍在聚类边界中，如果是，则将其也纳入于这个聚类，更新聚类中心和聚类边界。如果否，则将其纳入第二个特征聚类。
        3) 重复上述操作，直到扫描过所有m/z，
    """

    data = extract_mz_intensity_from_scm_events(mdata)
    data = filter_intensity(data, intensity_threshold)
    data = data.sort_values(by='mz', ascending=True).reset_index(drop=True)

    mdata.logger.info(f"Start metabolic feature extraction using heuristic clustering")
    mdata.logger.info(f"1) Cluster mz in ascending order according to heuristic clustering")
    feature_cluster = heuristic_clustering(data, ppm_threshold, min_cluster_size)

    mdata.logger.info(f"2) Use the method of kernel density estimation to find the density center of these Features.")
    density_centers = get_cluster_density_center_byKDE(feature_cluster)

    mdata.logger.info(f"3) Reconfirm the density centers of Features and merge the redundant density centers.")
    cluster_density_center = double_check_cluster_density_center(data, density_centers, ppm_threshold)

    mdata.logger.info(f"4) Perform feature clustering based on Feature mz (density center) values.")
    feature_cluster_result = extract_feature_by_mz_list(data, cluster_density_center)

    mdata.logger.info(f"5) Filter out redundant data results based on Feature clustering rules.")
    feature_cluster_result_filter_robust = filter_redundant_data(feature_cluster_result)

    mdata.logger.info(f"6) Construct a cell feature matrix based on the feature clustering results.")
    cell_feature_matrix = construct_cell_feature_matrix(feature_cluster_result_filter_robust)

    mdata.logger.info("Complete metabolic feature extraction using heuristic clustering.")
    mdata.cell_feature_matrix = cell_feature_matrix

    mdata.processing_status['feature_extraction_strategy'] = 'heuristic clustering'
    return mdata

def heuristic_clustering(data, ppm_threshold: float, min_cluster_size: int=3):
    """
    启发式聚类算法通过动态更新聚类中心和范围，对m/z数据进行分布聚类，提取代谢特征。
    聚类终止条件: 当纳入新的m/z时，聚类范围超过10ppm, 停止聚类
    实现步骤
    1) 筛选低强度信号并按照m/z值排序。剔除掉强度<500的m/z数据，减少噪音干扰。整合所有细胞的MS profiles并按照m/z值从小到大进行排列方便后续分析。
    2) 初始化第一个聚类。从最小的 m/z 开始，创建第一个聚类特征（Feature=1），记录 m/z 值、强度、来源细胞，并设置初始聚类密度中心。
    3) 判断是否属于当前聚类。计算当前聚类密度中心以(聚类所有m/z的平均值)及范围(聚类中心+-10ppm)，检查下一个m/z是否落入此范围：
    如果在范围内，则将其纳入当前聚类，更新聚类密度中心及范围。如果不在范围内，则将初始化一个新的聚类特征。
    4)重复判断和更新的过程，直到所有m/z分配完成。
    """

    data['Feature'] = 0
    feature_flag = 1
    pointer1 = 0

    mz_values = data['mz'].values
    progress_bar = tqdm(total=len(data))

    while pointer1 < len(data):
        pointer2 = pointer1
        mz_upper = mz_values[pointer1] * (1 + ppm_threshold * 1e-6)
        center_mz = mz_values[pointer1]

        while pointer2 < len(data) and mz_values[pointer2] <= mz_upper:
            pointer2 += 1

        while pointer2 < len(data) and mz_values[pointer2] > mz_upper:
            center_mz = mz_values[pointer1:pointer2].mean()
            mz_upper = center_mz * (1 + ppm_threshold * 1e-6)

            if mz_values[pointer2] <= mz_upper:
                pointer2 += 1
            else:
                break

        cluster_size = pointer2 - pointer1
        if cluster_size <= min_cluster_size:
            data.loc[pointer1:pointer2 - 1, 'Feature'] = 0
        else:
            data.loc[pointer1:pointer2 - 1, 'Feature'] = feature_flag
            feature_flag += 1

        pointer1 = pointer2
        progress_bar.update(pointer1-progress_bar.n)

    progress_bar.close()
    data = data[data['Feature'] != 0].reset_index(drop=True)

    return data

def get_cluster_density_center_byKDE(feature_cluster):
    """
    使用KDE方法为每个特征群计算密度中心。

    参数:
    ----------
    feature_cluster : pd.DataFrame
        包含分组数据的数据框，其中需要包含 'Feature' 和 'mz' 两列。

    返回:
    -------
    density_centers : list
        每个特征的密度中心值列表。
    """
    density_centers = []
    grouped = feature_cluster.groupby('Feature')

    with tqdm(total=len(grouped), desc='Identify feature cluster density center byKDE') as pbar:
        for feature, group in grouped:
            mz_list = group['mz']

            if len(set(mz_list)) <= 1:
                # 如果mz_list中有多个元素，但它们是相同的值,获取这个唯一值作为密度中心。
                density_center = next(iter(mz_list))
            else:
                density_center = calculate_density_center_byKDE(mz_list, range_number=10000)

            # add density_center to list
            density_centers.extend([density_center])

            # update pbar
            pbar.update(1)
        return density_centers

def calculate_density_center_byKDE(mz_list, range_number: int = 10000):
    """
    Description:
    ------------
    use kernel density estimation (KDE) to calculate the density,
    determine the density center within the range, and plot the KDE graph.

    Return:
    -------
    density_center(float)
    """

    # use kernel density estimation (KDE) to calculate the density.
    kde = gaussian_kde(mz_list.dropna().values)
    # create an array of mz values over a broad range for assessing density
    mz_range = np.linspace(mz_list.min(),mz_list.max(), range_number)
    # calculate the density for each point within this range.
    density_values = kde.evaluate(mz_range)
    density_center = mz_range[np.argmax(density_values)]
    return density_center

def double_check_cluster_density_center(mz_intensity, cluster_density_center, ppm_threshold: float = 10):
    """
    reconfirm the cluster density center, merge the cluster centers that are very close to or overlap with each other.
    """
    cluster_density_center.sort()
    result_df = pd.DataFrame(columns=['mz'])

    i = 0
    for i in tqdm(range(len(cluster_density_center) - 1), desc="Reconfirm the cluster density center."):
        m1 = cluster_density_center[i]
        m2 = cluster_density_center[i + 1]

        distance_condition = (m2 - m1) / m1 * 1e6 < ppm_threshold * 2
        if distance_condition:
            m3 = reconfirm_cluster_density_center(mz_intensity, m1, m2, ppm_threshold)
            # assign m3 to the next cluster density center.
            cluster_density_center[i + 1] = m3
        else:
            result_df = pd.concat([result_df, pd.DataFrame({'mz': [m1]})], ignore_index=True)

    # Add the last cluster density center to the result if it does not meet the distance condition.
    if i == len(cluster_density_center) - 1:
        result_df = pd.concat([result_df, pd.DataFrame({'mz': [cluster_density_center[-1]]})], ignore_index=True)

    return result_df

def reconfirm_cluster_density_center(mz_intensity, m1, m2, ppm_threshold: float=10):
    """
    reconfirm the density center of a cluster using Kernel Density Estimation (KDE).
    """
    # Calculate mz_lower and mz_upper based on the values of m1 and m2.
    mz_lower = m1 * (1 - 1e-6 * ppm_threshold)
    mz_upper = m2 * (1 + 1e-6 * ppm_threshold)
    # Filter out the rows that meet the criteria.
    selected_rows = mz_intensity[(mz_intensity['mz'] >= mz_lower) & (mz_intensity['mz'] <= mz_upper)]

    mz_list = selected_rows['mz']
    density_center = calculate_density_center_byKDE(mz_list)
    return density_center

def extract_feature_by_mz_list(mz_intensity, cluster_density_center, ppm_threshold: float=10):
    """
    Perform feature clustering based on fixed Feature mz (density center) values.
    """
    mz_list = cluster_density_center['mz']
    feature_cluster = mz_intensity
    mz_df = pd.DataFrame({'ref_mz': mz_list})
    mz_df[['mz_lower', 'mz_upper']] = mz_df['ref_mz'].apply(
        lambda mz: mz_threshold(mz, ppm_threshold)).apply(pd.Series)

    cluster_result = pd.DataFrame(
        columns = ['ref_mz', 'mz_lower', 'mz_upper', 'scan_Id', 'mz', 'intensity', 'CellNumber', 'Feature'])
    flag = 1
    for idx, row in tqdm(mz_df.iterrows(), total=len(mz_df), desc="Extract feature by mz list."):
        filtered = feature_cluster[(feature_cluster['mz'] >= row['mz_lower']) & (feature_cluster['mz'] <= row['mz_upper'])]
        filtered['Feature'] = flag
        filtered['ref_mz'] = row['ref_mz']
        filtered['mz_lower'] = row['mz_lower']
        filtered['mz_upper'] = row['mz_upper']
        cluster_result = pd.concat([cluster_result, filtered], ignore_index=True)
        flag += 1
    return cluster_result

def filter_redundant_data(cluster_result):
    """
    Filter out redundant data results based on Feature clustering rules.
    """
    # Step 1: Group by 'Feature' andd 'CellNumber', retain the row with the maximum intensity value in each group.
    filtered_rows_indices = []
    # Group by 'Feature', 'CellNumber'
    grouped = cluster_result.groupby(['Feature', 'CellNumber'])
    # handle single-row groups
    single_row_groups = grouped.filter(lambda x: len(x) == 1)
    filtered_rows_indices.extend(single_row_groups.index.tolist())
    # handle multi-row groups
    multi_row_groups = grouped.filter(lambda x: len(x) > 1)
    for _, group in tqdm(multi_row_groups.groupby(['Feature', 'CellNumber']), desc="Filter redundant data."):
        max_intensity_index = group['intensity'].idxmax()
        filtered_rows_indices.append(max_intensity_index)
    # extracted rows with maximum intensity values.
    filtered_rows = cluster_result.loc[filtered_rows_indices]
    return filtered_rows

def construct_cell_feature_matrix(cluster_result: pd.DataFrame):
    total_cells = len(cluster_result['CellNumber'].unique())
    cell_feature_matrix = pd.DataFrame(
        index=cluster_result['Feature'].unique(),
        columns=cluster_result['CellNumber'].unique())
    for index, row in tqdm(cluster_result.iterrows(), total=len(cluster_result),
                           desc="Construct a cell feature matrix based on clustering results."):
        cell_feature_matrix.at[row['Feature'], row['CellNumber']] = row['intensity']

    # Sort the matrix data by column names
    cell_feature_matrix = cell_feature_matrix.sort_index(axis=1)

    cluster_info = pd.DataFrame()
    for idx in tqdm(cell_feature_matrix.index, total=len(cell_feature_matrix.index), desc="Add cluster information to cell feature matrix."):
        cluster = cluster_result[cluster_result['Feature'] == idx]
        mz_mean = cluster['mz'].mean()
        mz_median = cluster['mz'].median()
        cellnumber = len(cluster['CellNumber'].unique())
        cellratio = cellnumber / total_cells * 100
        if 'metabolite' in cluster:
            info = {'Feature': idx, 'mz_center': cluster['ref_mz'].iloc[0], 'mz_mean': mz_mean,
                    'mz_median': mz_median, 'metabolite': cluster['metabolite'].iloc[0], 'hits': cellnumber, 'hit_rate': cellratio}
        else:
            info = {'Feature': idx, 'mz_center': cluster['ref_mz'].iloc[0], 'mz_mean': mz_mean,
                    'mz_median': mz_median, 'hits': cellnumber,'hit_rate': cellratio}

        cluster_info = pd.concat([cluster_info, pd.DataFrame([info])], ignore_index=True)

    cluster_info.set_index('Feature', inplace=True)
    cell_feature_matrix = cluster_info.join(cell_feature_matrix)
    cell_feature_matrix = cell_feature_matrix.reset_index().rename(columns={'index': 'Feature'})
    cell_feature_matrix.sort_values(by='hit_rate', ascending=False, inplace=True)

    return cell_feature_matrix
