import os
import scanpy as sc
from sklearn.cluster import KMeans

def tsne_scm(adata, result_path, show=False):
    sc.tl.pca(adata, n_comps=50)
    sc.pp.neighbors(adata, n_pcs=10)
    sc.tl.tsne(adata)

    sc.settings.figdir = result_path

    sc.pl.tsne(adata,
               color='TIC',
               cmap="coolwarm",
               title="tSNE - Raw Data Distribution (TIC)",
               show=show,
               save="_raw_data_distribution_TIC.pdf"
               )

    sc.pl.tsne(adata,
               color='scan_start_time',
               cmap="coolwarm",
               title="tSNE - Raw Data Distribution (Time)",
               show=show,
               save="_raw_data_distribution_Time.pdf"
               )

    sc.tl.leiden(adata, resolution=0.1)
    adata.obs['leiden_0.1'] = adata.obs['leiden']
    sc.pl.tsne(adata,
               color='leiden',
               title="tSNE - Raw Data Distribution (leiden)",
               show=show,
               save="_raw_data_distribution_leiden_0.1.pdf"
               )

    sc.tl.leiden(adata, resolution=0.5)
    adata.obs['leiden_0.5'] = adata.obs['leiden']
    sc.pl.tsne(adata,
               color='leiden',
               title="tSNE - Raw Data Distribution (leiden)",
               show=show,
               save="_raw_data_distribution_leiden_0.5.pdf"
               )

    X_pca = adata.obsm["X_pca"]
    kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto')
    adata.obs["kmeans"] = kmeans.fit_predict(X_pca).astype(str)
    sc.pl.tsne(adata,
               color='kmeans',
               title="tSNE - Raw Data Distribution (KMeans)",
               show=show,
               save="_raw_data_distribution_KMeans.pdf"
               )

    cell_tsne_info = adata.obs
    if 'cell_type_name' in cell_tsne_info.columns:
        cell_tsne_info = cell_tsne_info[
            ['scan_Id', 'scan_start_time', 'TIC', 'TIC_baselines', 'cell_type_name', 'leiden_0.1', 'leiden_0.5',
             'kmeans']]
    else:
        cell_tsne_info = cell_tsne_info[
            ['scan_Id', 'scan_start_time', 'TIC', 'TIC_baselines', 'leiden_0.1', 'leiden_0.5', 'kmeans']]

    cell_tsne_info.to_excel(f"{result_path}/cell_tsne_info.xlsx")


def tsne_cell_type(adata, result_path, show=False):
    sc.settings.figdir = result_path
    adata2 = adata.copy()
    sc.tl.pca(adata, n_comps=50)
    sc.pp.neighbors(adata, n_pcs=10)
    sc.tl.tsne(adata)

    unique_types = adata.obs[['cell_type_name', 'cell_type_color']].drop_duplicates()
    color_dict = dict(zip(unique_types['cell_type_name'], unique_types['cell_type_color']))

    sc.pl.tsne(adata,
               color='cell_type_name',
               title="tSNE - Raw Data Distribution (Cell type marker)",
               palette=color_dict,
               show=show,
               save="_raw_data_distribution_cell_type_marker_raw.pdf"
               )

    adata = adata[adata.obs["cell_type_name"] != "Unknown"].copy()
    adata = adata[adata.obs["cell_type_name"] != "Mixed"].copy()
    sc.pl.tsne(adata,
               color='cell_type_name',
               title="tSNE - Raw Data Distribution (Cell type marker)",
               palette=color_dict,
               show=show,
               save="_raw_data_distribution_cell_type_marker_clear.pdf"
               )

    clear_result_path = os.path.join(result_path, 'clear')
    if not os.path.exists(clear_result_path):
        os.makedirs(clear_result_path, exist_ok=True)
    adata2 = adata2[adata2.obs["cell_type_name"] != "Unknown"].copy()
    adata2 = adata2[adata2.obs["cell_type_name"] != "Mixed"].copy()

    tsne_scm(adata2, clear_result_path, show=False)

    sc.settings.figdir = clear_result_path
    sc.tl.pca(adata2, n_comps=50)
    sc.pp.neighbors(adata2, n_pcs=10)
    sc.tl.tsne(adata2)

    sc.pl.tsne(adata2,
               color='cell_type_name',
               title="tSNE - Raw Data Distribution (Cell type marker)",
               palette=color_dict,
               show=show,
               save="_raw_data_distribution_cell_type_marker_clear.pdf"
               )


def tsne_analysis(adata, result_path, hit_rate_threshold=0.2, show=False):
    output_path = os.path.join(result_path, 'tSNE')
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    tsne_scm(adata, output_path, show=show)
    if 'cell_type' in adata.obs:
        if 'cell_type_color' in adata.obs:
            tsne_cell_type(adata, output_path, show=show)
