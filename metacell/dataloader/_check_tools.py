import pandas as pd

def check_mz_conflicts_by_cell(result_df: pd.DataFrame, cell_number: str, ppm_threshold: float = 10):
    """
    检查指定 CellNumber 下相邻 m/z 差值是否小于指定的 ppm_threshold。

    参数：
    - result_df: 包含 'CellNumber', 'mz' 等信息的 DataFrame
    - cell_number: 指定的 CellNumber 值
    - ppm_threshold: ppm 差值阈值，默认10 ppm

    返回：
    - conflict_df: DataFrame，包含所有相邻 m/z 差值小于指定阈值的行
    """
    # 1. 提取指定 CellNumber 的行
    cell_df = result_df[result_df['CellNumber'] == cell_number].copy()

    # 2. 按 mz 排序
    cell_df = cell_df.sort_values(by='mz').reset_index(drop=True)

    # 3. 初始化一个列表来存储有问题的行
    conflict_rows = []

    # 4. 遍历相邻的 mz 值，检查 ppm 差值
    for i in range(1, len(cell_df)):
        mz_prev = cell_df.loc[i-1, 'mz']
        mz_curr = cell_df.loc[i, 'mz']

        ppm_diff = abs(mz_curr - mz_prev) / mz_prev * 1e6

        if ppm_diff < ppm_threshold:
            # 如果 ppm 差值小于阈值，将这两行加入冲突列表
            conflict_rows.append(cell_df.iloc[i-1])
            conflict_rows.append(cell_df.iloc[i])

    # 5. 如果有冲突行，合并成一个 DataFrame 并去重
    if conflict_rows:
        conflict_df = pd.DataFrame(conflict_rows).drop_duplicates().reset_index(drop=True)
    else:
        conflict_df = pd.DataFrame(columns=result_df.columns)

    return conflict_df

def plot_mz_hit_range(data: pd.DataFrame, target_mz: float, ppm_threshold: float=10):
    """
    Visualize the distribution of hits for all points within the ppm range of a given m/z.
    X axis: mz, Y axis: hits. Highlight the target m/z.
    Args:
        data (pd.DataFrame): DataFrame with 'mz' and 'hits' columns.
        target_mz (float): The target m/z value.
        ppm_threshold (float): ppm tolerance.
    """
    import matplotlib.pyplot as plt
    # Calculate ppm range
    mz_min = target_mz * (1 - ppm_threshold / 1e6)
    mz_max = target_mz * (1 + ppm_threshold / 1e6)
    subset = data[(data['mz'] >= mz_min) & (data['mz'] <= mz_max)].copy()
    plt.figure(figsize=(8, 5))
    plt.scatter(subset['mz'], subset['hits'], c='blue', label='In hit range', alpha=0.7)
    # Highlight the target mz point(s)
    target_points = subset[abs(subset['mz'] - target_mz) < 1e-8]
    if not target_points.empty:
        plt.scatter(target_points['mz'], target_points['hits'], c='red', label='Target m/z', s=60, marker='*')
    plt.axvline(target_mz, color='red', linestyle='--', label=f'target m/z={target_mz:.4f}')
    plt.xlabel('m/z')
    plt.ylabel('Hits')
    plt.title(f'Distribution of hits in ppm range of m/z={target_mz:.4f} (±{ppm_threshold} ppm)')
    plt.legend()
    plt.tight_layout()
    plt.show()