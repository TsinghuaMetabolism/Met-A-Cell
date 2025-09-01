

def fill_nan_with_half_min(data, start_col=8):
    subset = data.iloc[:, start_col - 1:]  # 提取指定列到最后一列
    min_values = subset.min(axis=1) / 2    # 计算每一行的最小值并取一半
    data.iloc[:, start_col - 1:] = subset.apply(lambda row: row.fillna(min_values[row.name]), axis=1)
    return data