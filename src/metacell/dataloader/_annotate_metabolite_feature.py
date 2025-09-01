import numpy as np

def annotate_metabolites_feature(mdata, metab_anno, ppm_threshold=10):
    """
    根据MAL_df的Theoretical_value列与cell_feature_matrix的mz_mean列的ppm差值，对cell_feature_matrix进行代谢物注释。

    参数：
    - cell_feature_matrix: 包含mz_mean列的DataFrame，需要被注释的特征矩阵。
    - metab_anno: 包含mz和metabolite	列的DataFrame，用于匹配的代谢物数据库。
    - ppm_threshold: ppm阈值，默认为10。如果mz_mean和mz之间的ppm差值小于该阈值，则认为匹配。

    返回：
    - 注释后的cell_feature_matrix，增加了Metabolites_name列。
    """
    # 提取需要的列为NumPy数组
    mz_mean_array = mdata.cell_feature_matrix['mz_mean'].values
    theoretical_values = metab_anno['mz'].values

    # 计算差值矩阵（绝对值）
    diff_matrix = np.abs(mz_mean_array[:, None] - theoretical_values[None, :])

    # 计算ppm差值矩阵
    ppm_matrix = diff_matrix / theoretical_values * 1e6

    # 找到满足ppm差值小于阈值的匹配
    matches = ppm_matrix < ppm_threshold

    # 为每个mz_mean收集匹配的Metabolites_name
    metabolite_names_list = []
    for i in range(len(mz_mean_array)):
        matched_indices = np.where(matches[i])[0]
        if len(matched_indices) == 0:
            metabolite_names_list.append('')
        else:
            matched_names = metab_anno.iloc[matched_indices]['metabolite'].tolist()
            # 用'/'连接多个匹配的名称
            names_str = '/'.join(matched_names)
            metabolite_names_list.append(names_str)

    # 将结果添加到cell_feature_matrix的Metabolites_name列
    mdata.cell_feature_matrix['metabolite'] = metabolite_names_list
    mdata.cell_feature_matrix.insert(4, 'metabolite', mdata.cell_feature_matrix.pop('metabolite'))

    return mdata
