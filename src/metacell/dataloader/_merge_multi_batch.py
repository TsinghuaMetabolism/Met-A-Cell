import numpy as np
import pandas as pd
import anndata as ad
from ._read_and_load import load_batch_and_group_data

def ppm_diff(x, y):
    """
    计算ppm差值: 绝对差除以较小值，再乘以1e6
    """
    return abs(x - y) / min(x, y) * 1e6


def process_batches(batch_data, metadata_columns):
    """处理 batch 数据，提取元数据和细胞数据"""
    batches = {}
    meta_list = []

    batch_list = list(batch_data.keys())  # 自动获取 batch_list

    for batch_name, df in batch_data.items():
        meta_cols = df.columns[:metadata_columns]
        cell_cols = df.columns[metadata_columns:]

        batches[batch_name] = {
            'df': df,
            'cell_cols': list(cell_cols),
            'n_cells': len(cell_cols)
        }

        meta_df = df[meta_cols].copy()
        meta_df['Batch'] = batch_name
        meta_df['orig_index'] = df.index
        meta_list.append(meta_df)

    combined_meta = pd.concat(meta_list, ignore_index=True)
    combined_meta['mz_mean'] = pd.to_numeric(combined_meta['mz_mean'])
    combined_meta = combined_meta.sort_values(by='mz_mean').reset_index(drop=True)

    return batches, combined_meta, batch_list

def group_by_mz(combined_meta):
    """根据 mz_mean 进行分组"""
    groups = []
    current_group = [combined_meta.iloc[0]]

    for i in range(1, len(combined_meta)):
        current_row = combined_meta.iloc[i]
        prev_row = current_group[-1]
        if ppm_diff(current_row['mz_mean'], prev_row['mz_mean']) < 10:
            current_group.append(current_row)
        else:
            groups.append(current_group)
            current_group = [current_row]

    groups.append(current_group)
    return groups

def merge_groups(groups, batches, batch_list, metadata_columns):
    """合并分组数据"""
    merged_rows = []
    total_cells_all = sum(batches[batch]['n_cells'] for batch in batch_list)

    for group in groups:
        group_df = pd.DataFrame(group)
        merged_mz = group_df['mz_mean'].mean()
        metabolite_name = 'Metabolites_name' if 'Metabolites_name' in group_df.columns else 'metabolite'
        count_name = 'CellNumber_count' if 'CellNumber_count' in group_df.columns else 'hits'
        ratio_name = 'CellNumber_ratio' if 'CellNumber_ratio' in group_df.columns else 'hit_rate'

        names = group_df[metabolite_name].dropna().unique().tolist()
        merged_name = "/".join(names)

        merged_count = {}
        merged_ratio = {}
        batch_presence = {}
        batch_counts = {}

        for batch in batch_list:
            sub = group_df[group_df['Batch'] == batch]
            if not sub.empty:
                count_val = str(sub.iloc[0][count_name])
                ratio_val = str(sub.iloc[0][ratio_name])
                merged_count[batch] = count_val
                merged_ratio[batch] = ratio_val
                batch_counts[batch] = float(sub.iloc[0][count_name]) if sub.iloc[0][count_name] else 0
                batch_presence[batch] = True
            else:
                merged_count[batch] = 'NaN'
                merged_ratio[batch] = 'NaN'
                batch_counts[batch] = 0
                batch_presence[batch] = False

        merged_count_str = "/".join([str(merged_count[b]) for b in batch_list])
        merged_ratio_str = "/".join([str(merged_ratio[b]) for b in batch_list])

        total_count = sum(batch_counts.values())
        total_ratio = total_count / total_cells_all * 100 if total_cells_all else np.nan
        batch_count = sum(1 for present in batch_presence.values() if present)

        cell_data_dict = {}
        for batch in batch_list:
            n_cells = batches[batch]['n_cells']
            cell_names = list(batches[batch]['df'].columns[metadata_columns:])
            cell_names = [f'{batch}_Cell{int(x[4:])}' for x in cell_names]
            cell_series = pd.Series([np.nan] * n_cells, index=cell_names)

            sub = group_df[group_df['Batch'] == batch]
            if not sub.empty:
                row = sub.iloc[0]
                orig_idx = int(row['orig_index'])
                df = batches[batch]['df']
                cell_vals = df.iloc[orig_idx, metadata_columns:].copy()
                cell_vals.index = cell_names
                cell_series = cell_vals

            cell_data_dict[batch] = cell_series

        merged_cell_data = pd.concat([cell_data_dict[b] for b in batch_list])

        merged_row = {
            'mz': merged_mz,
            'Metabolites_name': merged_name,
            'CellNumber_count': merged_count_str,
            'CellNumber_ratio': merged_ratio_str,
            'Total_Cell_Ratio': total_ratio,
            'Batch_Count': batch_count
        }
        merged_row.update(merged_cell_data.to_dict())
        merged_rows.append(merged_row)

    return merged_rows


def filter_batch_data(batch_data, group_data, metadata_columns=8):
    filtered_data = {}  # 创建一个空字典用于存储筛选后的数据

    for key in batch_data.keys():
        # 获取当前batch的原始数据
        batch_df = batch_data[key].copy()
        group_cell_numbers = set(group_data[key]['CellNumber'])

        # 获取前8列和满足条件的列
        retained_columns = list(batch_df.columns[:metadata_columns]) + [
            col for col in batch_df.columns[metadata_columns:] if col in group_cell_numbers
        ]

        # 筛选数据并存储
        filtered_data[key] = batch_df[retained_columns]

    return filtered_data

def merge_all_data(batch_dir, group_dir, metadata_columns=8):
    """主函数，执行完整流程"""
    batch_data, group_data = load_batch_and_group_data(batch_dir, group_dir)
    batch_data = filter_batch_data(batch_data, group_data, metadata_columns)
    batches, combined_meta, batch_list = process_batches(batch_data, metadata_columns)
    groups = group_by_mz(combined_meta)
    merged_data = merge_groups(groups, batches, batch_list, metadata_columns)
    merged_data = pd.DataFrame(merged_data)
    return batch_data, group_data, merged_data


def process_merged_data(merged_data, group_data, metadata_columns):
    """
    根据输入的 merged_data 分离细胞特征和特征元数据，并根据细胞名称构建细胞元数据 meta_cell。

    输入：
        merged_data: 包含特征元数据（前几列）和细胞数据（后续列）的 DataFrame。
        metadata_columns: 特征元数据所在的列数。
        group_data: 一个字典，键为批次名称（如 'Batch1'），值为对应批次的元数据 DataFrame，
                    其中每个 DataFrame 必须包含 'CellNumber', 'Group', 'TIC', 'scan_start_time', 'Kmean' 等列。
    输出：
        meta_cell: 根据细胞名称构建并合并了元数据的 DataFrame。
        meta_feature: 特征元数据（merged_data 的前 metadata_columns 列）。
        cell_feature: 细胞特征矩阵（即去掉特征元数据部分）。
    """
    # 前几列为特征的元数据，后续列为细胞数据
    feature_meta_cols = list(merged_data.columns[:metadata_columns])

    # 1. 细胞特征矩阵(去掉特征元数据部分)
    cell_feature = merged_data.drop(columns=feature_meta_cols)

    # 2. 特征元数据
    meta_feature = merged_data[feature_meta_cols].copy()

    # 3. 细胞元数据，根据cell_feature的列名生成
    cell_names = cell_feature.columns.tolist()
    meta_cell = pd.DataFrame({'Cell_Name': cell_names})

    # 根据 Cell_Name前缀获取批次列表
    batch_list = meta_cell['Cell_Name'].apply(lambda x: x.split('_')[0]).unique()

    # 第一轮：遍历批次列表
    for batch in batch_list:
        # 获取当前批次的细胞元数据
        batch_meta = group_data[batch]
        # 生成Cell_Name列，将例如Cell00001_Batch1 转换为Batch1_Cell1
        if 'Batch' in batch_meta.columns:
            # 当存在 Batch 列时，组合 Batch 列和 CellNumber 的数字部分
            batch_meta['Cell_Name'] = batch_meta.apply(
                lambda row: f"{row['Batch']}_Cell{int(row['CellNumber'][4:])}",
                axis=1
            )
        else:
            # 当不存在 Batch 列时，沿用原有逻辑处理
            batch_meta['Cell_Name'] = batch_meta['CellNumber'].apply(
                lambda x: f"{x.split('_')[1]}_Cell{int(x.split('_')[0][4:])}"
            )
        # 合并元数据到meta_cell中
        for idx, row in meta_cell.iterrows():
            cell_name = row['Cell_Name']
            if cell_name in batch_meta['Cell_Name'].values:
                cell_meta = batch_meta[batch_meta['Cell_Name'] == cell_name].iloc[0]
                meta_cell.loc[idx, ['Group', 'TIC', 'scan_start_time']] = cell_meta[
                    ['Group', 'TIC', 'scan_start_time']]

    return meta_cell, meta_feature, cell_feature


def store_as_anndata(meta_cell, meta_feature, cell_feature):
    """
    将meta_cell、meta_feature 和 cell_feature 转换为 anndata 格式。
    """
    meta_cell.index = meta_cell.index.astype(str)
    meta_feature.index = meta_feature.index.astype(str)
    # 创建 anndata 对象
    adata = ad.AnnData(
        X=cell_feature.values.T,
        obs=meta_cell,
        var=meta_feature,
    )
    adata.obs_names = [f"Cell_{i:d}" for i in range(adata.n_obs)]
    adata.var_names = [f"Feature_{i:d}" for i in range(adata.n_vars)]
    return adata