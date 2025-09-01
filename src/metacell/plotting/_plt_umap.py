import os
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.cluster import KMeans


def umap_scm(adata, result_path, show=False):
    # 执行PCA分析，从数据中提取最重要的主成分，计算前50个主成分
    sc.tl.pca(adata, n_comps=50)
    # sc.pl.pca_variance_ratio(adata, n_pcs=30, log=True)
    # 根据前10个主成分构建邻接图。计算细胞之间的相似性，以便后续的聚类、降维。
    sc.pp.neighbors(adata, n_pcs=10)
    # 执行UMAP降维。
    sc.tl.umap(adata)

    sc.settings.figdir = result_path
    # 绘制UMAP图
    # 依照TIC信息进行图注释
    sc.pl.umap(adata,
               color='TIC',
               cmap="coolwarm",
               title="UMAP - Raw Data Distribution (TIC)",
               show=show,
               save="_raw_data_distribution_TIC.pdf"
               )

    # 依照基线校正后的TIC信息进行图注释
    sc.pl.umap(adata,
               color='TIC_corrected',
               cmap="coolwarm",
               title="UMAP - Raw Data Distribution (TIC_corrected)",
               show=show,
               save="_raw_data_distribution_TIC_corrected.pdf"
               )

    # 依照scan_start_time进行图注释
    sc.pl.umap(adata,
               color='scan_start_time',
               cmap="coolwarm",
               title="UMAP - Raw Data Distribution (Time)",
               show=show,
               save="_raw_data_distribution_Time.pdf"
               )

    # 依照Leiden聚类结果进行图注释
    # Resolution=0.1
    sc.tl.leiden(adata, resolution=0.1)
    adata.obs['leiden_0.1'] = adata.obs['leiden']
    sc.pl.umap(adata,
               color='leiden',
               title="UMAP - Raw Data Distribution (leiden)",
               show=show,
               save="_raw_data_distribution_leiden_0.1.pdf"
               )

    # Resolution=0.5
    sc.tl.leiden(adata, resolution=0.5)
    adata.obs['leiden_0.5'] = adata.obs['leiden']
    sc.pl.umap(adata,
               color='leiden',
               title="UMAP - Raw Data Distribution (leiden)",
               show=show,
               save="_raw_data_distribution_leiden_0.5.pdf"
               )

    # 绘制UMAP图，按照kmeans信息进行图注释
    X_pca = adata.obsm["X_pca"]  # 提取 PCA 结果
    # 运行 K-means（n_clusters=2 表示 2 个簇）
    kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto')
    adata.obs["kmeans"] = kmeans.fit_predict(X_pca).astype(str)
    sc.pl.umap(adata,
               color='kmeans',
               title="UMAP - Raw Data Distribution (KMeans)",
               show=show,
               save="_raw_data_distribution_KMeans.pdf"
               )

    cell_umap_info = adata.obs.copy()
    umap_df = pd.DataFrame(adata.obsm['X_umap'], columns=['UMAP1','UMAP2'], index=adata.obs.index)
    cell_umap_info = pd.concat([cell_umap_info, umap_df], axis=1)

    # 判断是否有 'cell_type_name' 列，按需选择列
    if 'cell_type_name' in cell_umap_info.columns:
        cell_umap_info = cell_umap_info[
            ['scan_Id', 'scan_start_time', 'TIC', 'TIC_baselines', 'cell_type_name', 'leiden_0.1', 'leiden_0.5',
             'kmeans', 'UMAP1', 'UMAP2']]
    else:
        cell_umap_info = cell_umap_info[
            ['scan_Id', 'scan_start_time', 'TIC', 'TIC_baselines', 'leiden_0.1', 'leiden_0.5', 'kmeans', 'UMAP1', 'UMAP2']]

    cell_umap_info.to_excel(f"{result_path}/cell_umap_info.xlsx")


def umap_cell_type(adata, result_path, show=False):
    sc.settings.figdir = result_path
    # 执行PCA分析，从数据中提取最重要的主成分，计算前50个主成分
    adata2 = adata.copy()
    sc.tl.pca(adata, n_comps=50)
    # sc.pl.pca_variance_ratio(adata, n_pcs=30, log=True)
    # 根据前10个主成分构建邻接图。计算细胞之间的相似性，以便后续的聚类、降维。
    sc.pp.neighbors(adata, n_pcs=10)
    # 执行UMAP降维。
    sc.tl.umap(adata)

    # 制作颜色映射
    unique_types = adata.obs[['cell_type_name', 'cell_type_color']].drop_duplicates()
    color_dict = dict(zip(unique_types['cell_type_name'], unique_types['cell_type_color']))

    # 绘制原始的cell type marker注释的UMAP图谱
    sc.pl.umap(adata,
               color='cell_type_name',
               title="UMAP - Raw Data Distribution (Cell type marker)",
               palette=color_dict,
               # legend_loc='on data',
               show=show,
               save="_raw_data_distribution_cell_type_marker_raw.pdf"
               )

    # 绘制筛除Unknown以及Mixed后的cell type marker注释的UMAP图谱
    adata = adata[adata.obs["cell_type_name"] != "Unknown"].copy()
    adata = adata[adata.obs["cell_type_name"] != "Mixed"].copy()
    sc.pl.umap(adata,
               color='cell_type_name',
               title="UMAP - Raw Data Distribution (Cell type marker)",
               palette=color_dict,
               # legend_loc='on data',
               show=show,
               save="_raw_data_distribution_cell_type_marker_clear.pdf"
               )

    # 针对去除UnKnown以及Mixed之后的数据重新进行降维聚类分析。
    clear_result_path = os.path.join(result_path, 'clear')
    if not os.path.exists(clear_result_path):
        os.makedirs(clear_result_path, exist_ok=True)
    adata2 = adata2[adata2.obs["cell_type_name"] != "Unknown"].copy()
    adata2 = adata2[adata2.obs["cell_type_name"] != "Mixed"].copy()

    umap_scm(adata2, clear_result_path, show=False)

    sc.settings.figdir = clear_result_path
    sc.tl.pca(adata2, n_comps=50)
    sc.pp.neighbors(adata2, n_pcs=10)
    # 执行UMAP降维。
    sc.tl.umap(adata2)

    sc.pl.umap(adata2,
               color='cell_type_name',
               title="UMAP - Raw Data Distribution (Cell type marker)",
               palette=color_dict,
               # legend_loc='on data',
               show=show,
               save="_raw_data_distribution_cell_type_marker_clear.pdf"
               )

def umap_analysis(adata, result_path, show=False, use_TIC_correction=False):
    # 设置输出路径
    subfolder = 'UMAP_TIC_normalization' if use_TIC_correction else 'UMAP'
    output_path = os.path.join(result_path, subfolder)
    os.makedirs(output_path, exist_ok=True)

    # 始终计算 TIC_corrected
    if 'TIC_corrected' not in adata.obs.columns:
        if 'TIC' in adata.obs.columns and 'TIC_baselines' in adata.obs.columns:
            adata.obs['TIC_corrected'] = adata.obs['TIC'] - adata.obs['TIC_baselines']
        else:
            raise KeyError("Missing 'TIC' or 'TIC_baselines' in adata.obs")

    # 如果启用 TIC 校正，则对 adata.X 进行归一化
    if use_TIC_correction:
        tic_values = adata.obs['TIC_corrected'].values
        # 避免除以0
        tic_values = np.where(tic_values == 0, 1e-10, tic_values)
        adata.X = adata.X / tic_values[:, None]  # 每一行除以对应的 TIC 值

    umap_scm(adata, output_path, show=show)

    # 如果adata中含有cell_type信息时，则绘制cell_type注释的图谱
    if 'cell_type' in adata.obs:
        if 'cell_type_color' in adata.obs:
            umap_cell_type(adata, output_path, show=show)