import anndata as ad
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional


def plt_distribution(
    madata: ad.AnnData,
    axis: str,
    column: str,
    bins: int = 50,
    kde: bool = True,
    y_stat: str = "density",
    log_scale: bool = False,
    output_path: str | None = None
) -> None:
    """
    Plot an optimized distribution of a column in madata.obs or madata.var.

    Parameters
    ----------
    madata : ad.AnnData
        Input metabolomics AnnData object.
    column : str
        Column name in madata.obs or madata.var.
    axis : {"obs", "var"}, optional (default="obs")
        Specify whether to use madata.obs or madata.var.
    bins : int, optional (default=50)
        Number of histogram bins.
    kde : bool, optional (default=True)
        Whether to plot kernel density estimation (KDE) curve.
        Only applies if y_stat="density".
    y_stat : {"density", "count"}, optional (default="density")
        Whether to display density or raw counts on the y-axis.
    log_scale : bool, optional (default=False)
        Whether to use logarithmic scale on the x-axis.
    output_dir : str or None, optional (default=None)
        Output path for saving the figure as PDF. If None, show the figure instead.

    Returns
    -------
    None
    """
    # Validate inputs
    if axis not in {"obs", "var"}:
        raise ValueError("axis must be either 'obs' or 'var'")

    df = madata.obs if axis == "obs" else madata.var

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in madata.{axis}")

    if y_stat not in {"density", "count"}:
        raise ValueError("y_stat must be either 'density' or 'count'")

    data = df[column].dropna()

    color = "steelblue" if axis == "obs" else "darkorange"

    plt.figure(figsize=(6, 4))
    sns.histplot(
        data,
        bins=bins,
        kde=kde if y_stat == "density" else False,
        stat=y_stat,
        color=color,
        edgecolor="black",
        alpha=0.6,
        linewidth=0.5,
    )

    # KDE curve调整
    if kde and y_stat == "density":
        sns.kdeplot(
            data,
            color=color,
            linewidth=2,
            linestyle="--"
        )

    if log_scale:
        plt.xscale("log")

    plt.xlabel(column, fontsize=12)
    plt.ylabel(y_stat.capitalize(), fontsize=12)
    plt.title(f"Distribution of {column} ({axis})", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()

    if output_path is None:
        plt.show()
    else:
        plt.savefig(output_path, format="pdf")
        plt.close()

