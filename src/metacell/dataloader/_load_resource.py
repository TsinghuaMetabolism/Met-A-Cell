import pandas as pd
from pkg_resources import resource_filename

def load_meta_anno():
    # 获取meta_anno的文件路径
    file_path = resource_filename("meta_cell", "resource/meta_anno.xlsx")
    data = pd.read_excel(file_path)
    return data

