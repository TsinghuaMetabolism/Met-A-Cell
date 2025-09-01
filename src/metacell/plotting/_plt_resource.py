import matplotlib.pyplot as plt


def plt_cell_type_colors(df):
    """
    绘制细胞类型与颜色代码的示意图。

    参数:
        df (pd.DataFrame): 包含三列 ['cell_type', 'color_code', 'color_name']
    返回:
        显示一个 matplotlib 图像，表示每种细胞类型对应的颜色
    """
    if not all(col in df.columns for col in ['cell_type', 'color_code', 'color_name']):
        raise ValueError("DataFrame must contain 'cell_type', 'color_code', and 'color_name' columns.")

    fig, ax = plt.subplots(figsize=(8, len(df) * 0.6))

    for i, row in df.iterrows():
        ax.barh(i, 1, color=row['color_code'])
        label = f"{row['cell_type']}  |  {row['color_name']}  |  {row['color_code']}"
        ax.text(1.05, i, label, va='center', fontsize=10)

    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(0, 2)
    ax.set_frame_on(False)
    plt.title("Cell Type Color Legend", fontsize=14)
    plt.tight_layout()
    plt.show()