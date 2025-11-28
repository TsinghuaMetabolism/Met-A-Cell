import pandas as pd
from tqdm import tqdm
from ._utils import extract_mz_intensity_from_scm_events, filter_intensity, mz_threshold
from .scMetData import scMetData


def extract_features_by_metabolic_feature_library(mdata: scMetData, metab_anno: pd.DataFrame, intensity_threshold: float, ppm_threshold: float=10):
    """

    """
    data = extract_mz_intensity_from_scm_events(mdata)

    data = filter_intensity(data, intensity_threshold)

    cluster_result = feature_cluster_by_metabolic_feature_library(data, metab_anno, ppm_threshold)
    cluster_result_filter_robust = filter_redundant_features(cluster_result)

    cell_feature_matrix = construct_cell_feature_matrix(cluster_result_filter_robust)

    mdata.cell_feature_matrix = cell_feature_matrix

    mdata.processing_status['feature_extraction_strategy'] = 'metabolic feature library'
    return mdata


def feature_cluster_by_metabolic_feature_library(data: pd.DataFrame, metab_anno: pd.DataFrame, ppm_threshold: float=10):
    """
    通过代谢物特征数据库进行特征提取
    :param data: 单细细胞的MS profiles。
    :param metab_anno: 代谢物特征注释列表
    :param ppm_threshold: ppm threshold
    :return: 特征聚类结果。
    """
    cluster_result = pd.DataFrame(
        columns=['ref_mz', 'metabolite', 'mz_lower', 'mz_upper', 'scan_Id', 'mz', 'intensity', 'CellNumber', 'Feature']
    )
    flag = 1
    for idx, row in tqdm(metab_anno.iterrows(), total=len(metab_anno),
                         desc="Extract features by metabolic feature library."):
        mz = row['mz']
        name = row['metabolite']
        mz_lower, mz_upper = mz_threshold(mz, ppm_threshold)
        filtered = data[(data['mz'] >= mz_lower) & (data['mz'] <= mz_upper)]
        filtered['Feature'] = flag
        filtered['ref_mz'] = mz
        filtered['metabolite'] = name
        filtered['mz_lower'] = mz_lower
        filtered['mz_upper'] = mz_upper
        cluster_result = pd.concat([cluster_result, filtered], ignore_index=True)
        flag += 1

    return cluster_result


# === Filter out redundant data results based on feature clustering rules. ===
def filter_redundant_features(cluster_result: pd.DataFrame):
    # Step 1: Group by 'Feature' and 'CellNumber', retain the row with the maximum intensity value in each group
    filtered_rows_indices = []
    # Group by 'Feature', 'CellNumber'
    grouped = cluster_result.groupby(['Feature', 'CellNumber'])
    # Handle single-row groups
    single_row_groups = grouped.filter(lambda x: len(x) == 1)
    filtered_rows_indices.extend(single_row_groups.index.tolist())
    # Handle multi-row groups
    multi_row_groups = grouped.filter(lambda x: len(x) > 1)
    for _, group in tqdm(multi_row_groups.groupby(['Feature', 'CellNumber']), desc="Filtering robust rows"):
        max_intensity_index = group['intensity'].idxmax()
        filtered_rows_indices.append(max_intensity_index)
    # Extract rows with maximum intensity values
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

    return cell_feature_matrix




