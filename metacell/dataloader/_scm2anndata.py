import re
import numpy as np
import anndata as ad
import pandas as pd
from ..preprocessing import(
    rename_var_index,
)

def scm2anndata(mdata):
    cell_feature_matrix = mdata.cell_feature_matrix
    col_index = next((i for i, col in enumerate(cell_feature_matrix.columns) if re.match(r'^Cell\d+$', col)), None)
    madata = ad.AnnData(cell_feature_matrix.iloc[:,col_index:])
    madata.X = madata.X.astype(np.float64)

    madata.obs_names = cell_feature_matrix.iloc[:,col_index:].index
    madata.var_names = cell_feature_matrix.iloc[:,col_index:].columns

    obs_meta = cell_feature_matrix.iloc[:,:col_index]

    var_meta = mdata.scm_events[['CellNumber', 'scan_Id', 'scan_start_time', 'TIC']]
    if hasattr(mdata, "cell_marker") and isinstance(mdata.cell_marker, dict):
        for key in mdata.cell_marker.keys():
            if key == "TIC":
                continue  # 跳过 TIC
            candidate_cols = [f"{key}_mz", f"{key}_intensity"]
            for col in candidate_cols:
                if col in mdata.scm_events.columns and col not in var_meta.columns:
                    var_meta[col] = list(mdata.scm_events[col])
    var_meta['n_metabolites'] = np.sum(~np.isnan(madata.X), axis=0)
    var_meta = var_meta.set_index('CellNumber')
    var_meta['TIC_baselines'] = list(mdata.cell_marker_eic['TIC']['baselines'].iloc[mdata.scm_events_index[mdata.main_cell_marker]])
    var_meta['TIC_corrected'] = var_meta['TIC'] - var_meta['TIC_baselines']

    if isinstance(mdata.cell_type_marker_df, pd.DataFrame) and not mdata.cell_type_marker_df.empty:
        # 提取所需的列名
        marker_columns = [f"{marker}_mz" for marker in mdata.cell_type_marker_df['marker_name']] + \
                         [f"{marker}_intensity" for marker in mdata.cell_type_marker_df['marker_name']]

        # 确保所需列存在于 mdata.scm_events 中
        selected_columns = [col for col in marker_columns if col in mdata.scm_events.columns] + \
                           ['cell_type_encode', 'cell_type','cell_type_color','cell_type_name']

        # 将这些列添加到 var_meta
        for col in selected_columns:
            if col not in var_meta.columns:
                var_meta[col] = list(mdata.scm_events[col])

    madata = ad.AnnData(madata.X, obs=obs_meta, var=var_meta)
    madata.uns['logging'] = mdata.memory_handler.log_messages
    madata.uns['processing_status'] = mdata.get_processing_status()
    madata.obs = madata.obs.astype(str)
    madata.var = madata.var.astype(str)
    madata = madata.T
    cell_marker_list = list(mdata.cell_marker.keys())
    cell_marker = [f"{marker}_mz" for marker in cell_marker_list] + [f"{marker}_intensity" for marker in cell_marker_list]
    if isinstance(mdata.cell_type_marker_df, pd.DataFrame) and not mdata.cell_type_marker_df.empty:
        cols_to_convert = ['scan_start_time', 'TIC', 'TIC_baselines', 'TIC_corrected','n_metabolites', 'cell_type_encode'] + cell_marker
    else:
        cols_to_convert = ['scan_start_time', 'TIC', 'TIC_baselines', 'TIC_corrected','n_metabolites'] + cell_marker
    for col in cols_to_convert:
        madata.obs[col] = pd.to_numeric(madata.obs[col], errors='coerce')  # 转换为数值，无法转换的设为 NaN

    # 获取需要转换的列（排除 'metabolite'）
    cols_to_convert = [col for col in madata.var.columns if col != 'metabolite']
    # 批量转换为数值型
    for col in cols_to_convert:
        madata.var[col] = pd.to_numeric(madata.var[col], errors='coerce')  # 无法转换的值设为 NaN

    madata = rename_var_index(madata, col="mz_center", decimals=3, prefix="mz_")

    return madata