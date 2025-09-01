import os
import glob
import pandas as pd

def load_batch_and_group_data(batch_dir: str, group_dir: str) -> tuple[dict, dict]:
    """
    读取 Batch 文件夹中的 CSV 或 Excel 文件和 Group 文件夹中的 CSV 或 Excel 文件。

    参数：
        - batch_dir (str): Batch 文件夹路径，包含 CSV 或 Excel 文件。
        - group_dir (str): Group 文件夹路径，包含 CSV 或 Excel 文件。

    返回：
        - batch_data (dict): 键为批次名（例如 'Batch1'），值为对应的 DataFrame。
        - group_data (dict): 键为批次名（例如 'Batch1'），值为对应的 DataFrame。
    """
    batch_data, group_data = {}, {}

    # 加载 Batch 文件夹中的文件
    for file_path in glob.glob(os.path.join(batch_dir, '*')):
        if not os.path.isfile(file_path):
            continue

        file_ext = os.path.splitext(file_path)[1].lower()
        batch_key = os.path.splitext(os.path.basename(file_path))[0]  # 提取文件名（不带扩展名）

        if file_ext == '.csv':
            batch_data[batch_key] = pd.read_csv(file_path)
        elif file_ext in ['.xlsx', '.xls']:  # 兼容 xlsx 和 xls
            batch_data[batch_key] = pd.read_excel(file_path)

    # 加载 Group 文件夹中的文件
    for file_path in glob.glob(os.path.join(group_dir, '*')):
        if not os.path.isfile(file_path):
            continue

        file_ext = os.path.splitext(file_path)[1].lower()
        file_name = os.path.basename(file_path)
        batch_key = file_name.replace("Group info_", "").replace(file_ext, "")

        if file_ext == '.csv':
            group_data[batch_key] = pd.read_csv(file_path)
        elif file_ext in ['.xlsx', '.xls']:
            group_data[batch_key] = pd.read_excel(file_path)

    return batch_data, group_data