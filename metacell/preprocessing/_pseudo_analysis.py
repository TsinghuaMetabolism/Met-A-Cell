import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap

def plot_branch_celltype_composition(
    df,
    branch_col='branch',
    celltype_col='cell_type',
    show_internal_labels=True,
    perc_threshold=0.03,
    figsize=(3.5, 6),
    bar_width=0.5,
    font_family=('Helvetica', 'DejaVu Sans', 'Liberation Sans'),
    base_fontsize=11,
    save_path=None
):
    """
    绘制按 branch 聚合的 cell_type 组成的堆叠柱状图（归一化为比例），
    可选是否在柱内显示百分比注释，顶部显示总细胞数。
    优化视觉样式（Nature风格）。
    """

    # ------------------- 基本设置 -------------------
    #rcParams['font.family'] = 'sans-serif'
    #rcParams['font.sans-serif'] = list(font_family)
    #rcParams['font.size'] = base_fontsize
    #rcParams['axes.linewidth'] = 0.6

    # 统计比例
    count_df_raw = (
        df.groupby([branch_col, celltype_col])
        .size()
        .reset_index(name='count')
        .pivot(index=branch_col, columns=celltype_col, values='count')
        .fillna(0)
    )
    count_df = count_df_raw.div(count_df_raw.sum(axis=1), axis=0)

    # ------------------- 配色与顺序 -------------------
    color_map = {'LK': '#0050ff', 'LSK': '#1ac3aa', 'CLP': '#ff2a2a'}
    kept_cols = [c for c in color_map if c in count_df.columns]
    count_df = count_df[kept_cols]
    count_df_raw = count_df_raw[kept_cols]

    branch_order = ['pre_branch', 'branc_LK', 'branch_CLP']
    branch_rename = {'pre_branch': 'Pre-branch', 'branc_LK': 'LK branch', 'branch_CLP': 'CLP branch'}
    present_order = [b for b in branch_order if b in count_df.index]
    if not present_order:
        present_order = list(count_df.index)

    count_df = count_df.loc[present_order].rename(index=branch_rename)
    count_df_raw = count_df_raw.loc[present_order].rename(index=branch_rename)

    # ------------------- 绘图 -------------------
    fig, ax = plt.subplots(figsize=figsize)
    count_df.plot(
        kind='bar', stacked=True,
        color=[color_map[c] for c in count_df.columns],
        ax=ax, width=bar_width, edgecolor='none'
    )

    # 柱内百分比标注（可选）
    if show_internal_labels:
        for i, branch in enumerate(count_df.index):
            bottom = 0
            for c in count_df.columns:
                val = count_df.loc[branch, c]
                if val > perc_threshold:
                    ax.text(
                        i, bottom + val/2, f"{val*100:.0f}%",
                        ha='center', va='center',
                        fontsize=base_fontsize-2, color='white', fontweight='bold'
                    )
                bottom += val

    # 柱顶总细胞数
    for i, branch in enumerate(count_df_raw.index):
        total_n = int(count_df_raw.loc[branch].sum())
        ax.text(
            i, 1.04, f"n={total_n}", ha='center', va='bottom',
            fontsize=base_fontsize, color='#333333', fontweight='bold'
        )

    # ------------------- 视觉优化 -------------------
    ax.set_xlabel('Branch', fontsize=base_fontsize+1, labelpad=6)
    ax.set_ylabel('Proportion (%)', fontsize=base_fontsize+1, labelpad=8)
    ax.tick_params(axis='x', rotation=0, labelsize=base_fontsize, colors='#222222', pad=4)
    ax.tick_params(axis='y', labelsize=base_fontsize-1, colors='#444444', width=0.6)
    ax.set_ylim(0, 1.15)

    # 仅保留底部与左侧轴线（浅灰细线）
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_color('#444444')
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_color('#444444')

    ax.grid(False)
    ax.legend().remove()
    ax.set_facecolor('none')
    fig.patch.set_alpha(0.0)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=600, bbox_inches='tight', transparent=True)
    else:
        plt.show()

    return fig, ax
    

def plot_celltype_branch_composition(
    df,
    celltype_col='cell_type',
    branch_col='branch',
    show_internal_labels=True,
    show_legend=True,
    perc_threshold=0.03,
    figsize=(3.5, 6),
    bar_width=0.5,
    font_family=('Helvetica', 'DejaVu Sans', 'Liberation Sans'),
    base_fontsize=11,
    celltype_order=('LSK', 'LK', 'CLP'),
    save_path=None
):
    """
    以 cell_type 为分组，展示 branch 组成的堆叠柱状图（按 cell_type 行归一化为比例）。
    - 顶部显示每个 cell_type 的总细胞数 (n)
    - show_internal_labels: 是否在柱内显示百分比
    - show_legend: 是否显示图例（放在右侧）
    - 视觉风格：浅灰细轴、干净、出版级

    Parameters
    ----------
    df : pd.DataFrame
        包含 celltype_col 与 branch_col 的数据框。
    celltype_col : str
        细胞类型列名（默认 'cell_type'）。
    branch_col : str
        分支列名（默认 'branch'）。
    show_internal_labels : bool
        是否在柱内显示百分比注释（默认 True）。
    show_legend : bool
        是否显示图例，位置在右侧（默认 True）。
    perc_threshold : float
        百分比注释阈值（段占比 > 该值才显示，默认 0.03）。
    figsize : tuple
        画布尺寸（默认 (3.8, 6)）。
    bar_width : float
        柱子宽度（默认 0.5）。
    font_family : tuple
        字体优先级（避免 Arial 未安装告警）。
    base_fontsize : int
        全局基础字号（默认 11）。
    celltype_order : tuple or list
        x 轴上 cell_type 的顺序（默认 ('LSK','LK','CLP')，仅保留存在者）。
    save_path : str or None
        若提供路径，将保存到该文件（如 'figure.pdf'），否则直接显示。

    Returns
    -------
    (fig, ax)
    """

    # ------------------- 基本设置 -------------------
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = list(font_family)
    rcParams['font.size'] = base_fontsize
    rcParams['axes.linewidth'] = 0.6

    # 统计计数与比例（index=cell_type, columns=branch）
    count_df_raw = (
        df.groupby([celltype_col, branch_col])
          .size()
          .reset_index(name='count')
          .pivot(index=celltype_col, columns=branch_col, values='count')
          .fillna(0)
    )
    # 归一化为比例（按 cell_type 行）
    count_df = count_df_raw.div(count_df_raw.sum(axis=1), axis=0)

    # ------------------- 分支顺序、重命名与配色 -------------------
    # 统一三类分支及更易读名称
    branch_order = ['pre_branch', 'branc_LK', 'branch_CLP']
    branch_rename = {'pre_branch': 'Pre-branch', 'branc_LK': 'LK branch', 'branch_CLP': 'CLP branch'}
    # 颜色：与语义对齐（pre=灰、LK=蓝、CLP=红）
    branch_colors = {'Pre-branch': '#1ac3aa', 'LK branch': '#0000ff', 'CLP branch': '#ff0001'}

    # 仅保留出现的分支并排序
    present_branch = [b for b in branch_order if b in count_df.columns]
    count_df = count_df[present_branch]
    count_df_raw = count_df_raw[present_branch]

    # 重命名列用于图例显示
    count_df = count_df.rename(columns=branch_rename)
    count_df_raw = count_df_raw.rename(columns=branch_rename)
    kept_cols = [branch_rename[b] for b in present_branch]
    color_list = [branch_colors[c] for c in kept_cols]

    # ------------------- x 轴 cell_type 顺序 -------------------
    if celltype_order is not None:
        present_celltypes = [ct for ct in celltype_order if ct in count_df.index]
        if present_celltypes:
            count_df = count_df.loc[present_celltypes]
            count_df_raw = count_df_raw.loc[present_celltypes]

    # ------------------- 绘图 -------------------
    fig, ax = plt.subplots(figsize=figsize)
    count_df.plot(
        kind='bar', stacked=True,
        color=color_list,
        ax=ax, width=bar_width, edgecolor='none'
    )

    # 柱内百分比标注（可选）
    if show_internal_labels:
        for i, ct in enumerate(count_df.index):
            bottom = 0.0
            for col in kept_cols:
                val = count_df.loc[ct, col]
                if val > perc_threshold:
                    ax.text(
                        i, bottom + val/2, f"{val*100:.0f}%",
                        ha='center', va='center',
                        fontsize=base_fontsize-2, color='white', fontweight='bold'
                    )
                bottom += val

    # 顶部总数 (n)
    for i, ct in enumerate(count_df_raw.index):
        total_n = int(count_df_raw.loc[ct].sum())
        ax.text(
            i, 1.04, f"n={total_n}", ha='center', va='bottom',
            fontsize=base_fontsize, color='#333333', fontweight='bold'
        )

    # ------------------- 视觉优化 -------------------
    ax.set_xlabel('Cell type', fontsize=base_fontsize+1, labelpad=6)
    ax.set_ylabel('Proportion (%)', fontsize=base_fontsize+1, labelpad=8)
    ax.tick_params(axis='x', rotation=0, labelsize=base_fontsize, colors='#222222', pad=4)
    ax.tick_params(axis='y', labelsize=base_fontsize-1, colors='#444444', width=0.6)
    ax.set_ylim(0, 1.15)

    # 轴线：仅保留左、下（浅灰细线）
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines['bottom'].set_visible(True); ax.spines['bottom'].set_color('#444444')
    ax.spines['left'].set_visible(True);   ax.spines['left'].set_color('#444444')

    ax.grid(False)
    ax.set_facecolor('none')
    fig.patch.set_alpha(0.0)

    # 图例（右侧）
    if show_legend:
        leg = ax.legend(
            title='Branch',
            labels=kept_cols,
            frameon=False,
            loc='upper left',
            bbox_to_anchor=(1.02, 1.0),  # 右侧
            borderaxespad=0,
            fontsize=base_fontsize-1,
            title_fontsize=base_fontsize
        )
    else:
        ax.legend().remove()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=600, bbox_inches='tight', transparent=True)
    else:
        plt.show()

    return fig, ax


def build_dot_agg(df, sig_df, branch_levels, bin_step=1, value_mode="zscore",
                  min_positive=0.0, top_n=None, agg="mean"):
    """
    构建 Scanpy 风格 dotplot 需要的聚合表：
    - 颜色：每 bin×代谢物 的平均强度（z-score 或 raw）
    - 大小：每 bin×代谢物 的表达比例（>0 占比）
    """
    # 1) 显著代谢物且存在于列
    met_keep = sig_df.index[sig_df["pval_adj"] < 0.05].intersection(df.columns)
    if len(met_keep) == 0:
        raise ValueError("没有在数据表中找到 pval_adj<0.05 的代谢物列。")

    # 2) 过滤分支与必需列
    sub = df.loc[df["branch"].isin(branch_levels),
                 ["Pseudotime", "branch", *met_keep]].copy()
    sub = sub.dropna(subset=["Pseudotime"])
    if sub.empty:
        raise ValueError(f"分支 {branch_levels} 中没有数据。")

    # 3) 拟时序分箱（步长=1）
    sub["pseudotime_bin"] = (np.floor(sub["Pseudotime"] / bin_step) * bin_step).astype(int)

    # 4) 计算两类指标
    long = sub.melt(id_vars=["pseudotime_bin"],
                    value_vars=met_keep,
                    var_name="metabolite", value_name="value")

    # 表达比例（>min_positive 视为表达）
    frac_df = (long
               .assign(expr=lambda x: (x["value"] > min_positive).astype(float))
               .groupby(["pseudotime_bin", "metabolite"])["expr"]
               .mean()
               .rename("frac")
               .reset_index())

    # 强度聚合
    if agg == "median":
        stat = long.groupby(["pseudotime_bin","metabolite"])["value"].median()
    else:
        stat = long.groupby(["pseudotime_bin","metabolite"])["value"].mean()
    stat_df = stat.rename("stat").reset_index()

    # 合并
    agg_df = pd.merge(stat_df, frac_df, on=["pseudotime_bin","metabolite"], how="outer")

    # 5) 值模式：z-score or raw（按代谢物独立标准化）
    if value_mode == "zscore":
        def zscore(x):
            m = np.nanmean(x)
            s = np.nanstd(x)
            return (x - m) / s if s > 0 else np.zeros_like(x)
        agg_df["color_val"] = (agg_df.groupby("metabolite")["stat"]
                               .transform(lambda s: pd.Series(zscore(s.to_numpy()), index=s.index)))
        color_label = "z-score"
    else:
        agg_df["color_val"] = agg_df["stat"]
        color_label = "Intensity"

    # 6) 代谢物排序：按“峰值出现的最早 bin”
    order_tbl = (agg_df.sort_values(["metabolite","color_val"], ascending=[True, False])
                      .groupby("metabolite").first().reset_index()
                      .sort_values("pseudotime_bin"))
    ordered_mets = order_tbl["metabolite"].tolist()
    if top_n is not None and top_n > 0:
        ordered_mets = ordered_mets[:top_n]
        agg_df = agg_df[agg_df["metabolite"].isin(ordered_mets)]

    # 7) 补全所有 bin×代谢物（使图是完整网格）
    bins = np.sort(agg_df["pseudotime_bin"].unique())
    grid = pd.MultiIndex.from_product([bins, ordered_mets], names=["pseudotime_bin","metabolite"])
    agg_df = agg_df.set_index(["pseudotime_bin","metabolite"]).reindex(grid).reset_index()

    # 8) 返回数据与元信息
    return agg_df, {
        "bins": bins.tolist(),
        "metabolites": ordered_mets,
        "color_label": color_label
    }


def plot_metabolite_trend_single_branch(
    agg_df, info, title,
    base_width=0.5,   # 控制基因间距：越大越稀疏
    figsize_y=7, dpi=300,
    savepath=None,
    size_min=10, size_max=200,
    jitter=0.1        # 防止点完全重叠的随机微移
):
    """
    Nature 风格散点热图，横轴为基因（带间距与防重叠），纵轴为 Pseudotime。
    lk_agg, lk_info = build_dot_agg(
    df=young_hsc6_branch,
    sig_df=sig_young_hsc6,
    branch_levels=["pre_branch", "branc_LK"],
    bin_step=1,
    value_mode="zscore",   # 或 'raw'
    min_positive=0.0,      # >0 视为表达；若你定义“表达”为 > 某阈值，可调
    top_n=None,            # 太多代谢物时可设 eg. 60
    agg="mean"             # 或 'median'
    )
    plot_metabolite_trend_single_branch(
        lk_agg, lk_info,
        title="Metabolite expression dynamics along the LK branch",
        savepath=f"{result_path}/YOUNG-HSC6-LK_branch_metabolite_expression_dynamics.pdf"
    )

    """
    mets = info["metabolites"]
    bins = info["bins"]

    # 坐标映射
    x_pos = {m: i for i, m in enumerate(mets)}
    y_pos = {b: i for i, b in enumerate(bins)}

    xs = agg_df["metabolite"].map(x_pos).to_numpy(dtype=float)
    ys = agg_df["pseudotime_bin"].map(y_pos).to_numpy(dtype=float)
    color_vals = agg_df["color_val"].to_numpy()
    frac_vals = agg_df["frac"].fillna(0).to_numpy()

    # 随机轻微抖动，避免重叠
    #rng = np.random.default_rng(42)
    #xs = xs + rng.uniform(-jitter, jitter, size=len(xs))

    # 点大小按表达比例线性映射
    if np.nanmax(frac_vals) > 0:
        sizes = size_min + (size_max - size_min) * (frac_vals / np.nanmax(frac_vals))
    else:
        sizes = np.full_like(frac_vals, size_min, dtype=float)

    # Nature-style 蓝白红渐变
    colors = [
        (0.231, 0.298, 0.752),  # 柔和蓝 (#3b4cc0)
        (1.000, 1.000, 1.000),  # 白色
        (0.706, 0.016, 0.150)   # 柔和红 (#b40426)
    ]
    nature_cmap = LinearSegmentedColormap.from_list("nature_cmap", colors, N=256)

    # 自动调整画布宽度：基因越多，图越宽
    figsize_x = max(6, len(mets) * base_width)
    fig, ax = plt.subplots(figsize=(figsize_x, figsize_y), dpi=dpi)

    # 绘制散点
    sc = ax.scatter(
        xs, ys,
        c=color_vals,
        s=sizes,
        cmap=nature_cmap,
        edgecolors="none",
        alpha=0.9
    )

    # 设置坐标轴
    ax.set_xlim(-0.5, len(mets) - 0.5)
    ax.set_ylim(-0.5, len(bins) - 0.5)
    ax.set_xticks(range(len(mets)))
    ax.set_xticklabels(mets, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(bins)))
    ax.set_yticklabels(bins, fontsize=9)

    ax.set_xlabel("Metabolites (significant, p_adj < 0.05)", fontsize=11)
    ax.set_ylabel("Pseudotime", fontsize=11)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")

    # 边框和刻度优化
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=1.2, length=4)

    # 优化 colorbar
    cbar = plt.colorbar(sc, ax=ax, pad=0.015, aspect=30)
    cbar.set_label(info.get("color_label", "Expression (z-score)"), fontsize=10)
    cbar.ax.tick_params(labelsize=9, width=1.0)
    cbar.outline.set_linewidth(0.8)
    cbar.outline.set_edgecolor("gray")

    # 样式微调
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def _fit_slope_per_gene(df: pd.DataFrame, stat_col: str = "stat"):
    """对每个 metabolite 用 np.polyfit( bin, stat, 1 ) 求斜率；stat 不存在就用 color_val。"""
    if stat_col not in df.columns:
        stat_col = "color_val"
    slopes = {}
    for g, sub in df.groupby("metabolite"):
        x = sub["pseudotime_bin"].to_numpy(dtype=float)
        y = sub[stat_col].to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() >= 2:
            slope = np.polyfit(x[ok], y[ok], 1)[0]
        else:
            slope = np.nan
        slopes[g] = slope
    return slopes

def build_metabolite_order_by_direction(
    lk_agg: pd.DataFrame,
    clp_agg: pd.DataFrame,
    sig_df: pd.DataFrame,
    sig_col: str = "pval_sig"  # 若不存在则自动回退到 pval_adj
):
    """
    返回 final_order（左=CLP升高，右=LK升高；右侧从右到左p增大）和左右两侧集合。
    """
    # 共同代谢物
    common = sorted(set(lk_agg["metabolite"]).intersection(set(clp_agg["metabolite"])))
    if not common:
        raise ValueError("LK 与 CLP 无共同代谢物。")

    # 两边斜率
    lk_slopes  = _fit_slope_per_gene(lk_agg,  stat_col="stat")
    clp_slopes = _fit_slope_per_gene(clp_agg, stat_col="stat")

    # 选择用于排序的显著性列
    if sig_col not in sig_df.columns:
        sig_col = "pval_adj"
    # 取 p 值的 helper（缺失给一个大值，防止排在前面）
    def p_of(g):
        try:
            return float(sig_df.loc[g, sig_col])
        except Exception:
            return np.inf

    left_clp, right_lk = [], []
    for g in common:
        lk_s  = lk_slopes.get(g, np.nan)
        clp_s = clp_slopes.get(g, np.nan)

        # 判定规则：把“在该分支斜率>0 且更大”的基因分给对应分支；
        # 若两边都<=0，则不放；若都>0，就放到斜率更大的分支。
        if np.isfinite(clp_s) and clp_s > 0 and (not np.isfinite(lk_s) or clp_s >= lk_s):
            left_clp.append(g)
        elif np.isfinite(lk_s) and lk_s > 0 and (not np.isfinite(clp_s) or lk_s > clp_s):
            right_lk.append(g)
        # 否则（两边都不升高），跳过

    # 左侧：CLP 升高，小p在更左
    left_sorted  = sorted(left_clp,  key=lambda g: p_of(g))
    # 右侧：LK 升高，小p也在靠外（最右），因此对升序列表做反转以摆放
    right_sorted = sorted(right_lk, key=lambda g: p_of(g))

    final_order = left_sorted + right_sorted[::-1]
    return final_order, left_sorted, right_sorted

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd

def plot_metabolite_trend_dual_branch(
    lk_agg: pd.DataFrame, lk_info: dict,
    clp_agg: pd.DataFrame, clp_info: dict,
    sig_df: pd.DataFrame,
    title="Pseudotime-resolved metabolite expression (LK up, CLP down)",
    base_width=0.55, figsize_y=8, dpi=300, savepath=None,
    size_min=10, size_max=200, center_zero=True,
    custom_order: list[str] | None = None,
    sig_col: str = "pval_sig"
):
    """
    # 先根据方向+显著性生成顺序
    final_order, left_clp, right_lk = build_metabolite_order_by_direction(
        lk_agg, clp_agg, sig_young_hsc6, sig_col="pval_sig"  # 若 sig_df 没有 pval_sig 会自动退回 pval_adj
    )

    # 然后按该顺序出图
    plot_metabolite_trend_dual_branch(
        lk_agg=lk_agg, lk_info=lk_info,
        clp_agg=clp_agg, clp_info=clp_info,
        sig_df=sig_young_hsc6,
        custom_order=final_order,   # ← 关键：应用自定义顺序
        title="Metabolite expression along pseudotime (LK upward, CLP downward)",
        savepath=f"{result_path}/YOUNG-HSC6-metabolite_trend_dual_branch.pdf"
    )

    """
    # —— 前半部分：数据准备（与你上一版一致） ——
    lk_mets = set(lk_info["metabolites"])
    clp_mets = set(clp_info["metabolites"])
    common_mets = [m for m in (custom_order or lk_info["metabolites"]) if m in lk_mets & clp_mets]
    if not common_mets:
        raise ValueError("LK 与 CLP 无共同代谢物或 custom_order 为空。")

    lk_df  = lk_agg[lk_agg["metabolite"].isin(common_mets)].copy()
    clp_df = clp_agg[clp_agg["metabolite"].isin(common_mets)].copy()
    lk_df["metabolite"]  = pd.Categorical(lk_df["metabolite"],  categories=common_mets, ordered=True)
    clp_df["metabolite"] = pd.Categorical(clp_df["metabolite"], categories=common_mets, ordered=True)

    use_col = "stat" if "stat" in lk_df.columns else "color_val"
    comb = pd.concat([
        lk_df[["metabolite","pseudotime_bin",use_col]].assign(branch="LK"),
        clp_df[["metabolite","pseudotime_bin",use_col]].assign(branch="CLP")
    ], ignore_index=True)
    def zscore(x): 
        m, s = np.nanmean(x), np.nanstd(x)
        return (x-m)/s if s>0 else np.zeros_like(x)
    comb["color_val_norm"] = comb.groupby("metabolite")[use_col].transform(zscore)
    lk_df  = lk_df.merge(comb[comb["branch"]=="LK"][["metabolite","pseudotime_bin","color_val_norm"]],
                         on=["metabolite","pseudotime_bin"], how="left")
    clp_df = clp_df.merge(comb[comb["branch"]=="CLP"][["metabolite","pseudotime_bin","color_val_norm"]],
                         on=["metabolite","pseudotime_bin"], how="left")

    x_pos = {m:i for i,m in enumerate(common_mets)}
    lk_x, clp_x = lk_df["metabolite"].map(x_pos).to_numpy(float), clp_df["metabolite"].map(x_pos).to_numpy(float)
    lk_bins = np.sort(lk_df["pseudotime_bin"].unique()); clp_bins = np.sort(clp_df["pseudotime_bin"].unique())
    lk_y, clp_y = lk_df["pseudotime_bin"].to_numpy(float), -clp_df["pseudotime_bin"].to_numpy(float)

    frac_all = np.concatenate([lk_df["frac"].fillna(0).to_numpy(), clp_df["frac"].fillna(0).to_numpy()])
    sizes_all = (size_min + (size_max - size_min) * (frac_all / np.nanmax(frac_all))) if np.nanmax(frac_all)>0 else np.full_like(frac_all, size_min)
    lk_sizes, clp_sizes = sizes_all[:len(lk_df)], sizes_all[len(lk_df):]

    colors = [(0.231,0.298,0.752),(1,1,1),(0.706,0.016,0.150)]
    cmap = LinearSegmentedColormap.from_list("nature", colors, N=256)
    color_all = np.concatenate([lk_df["color_val_norm"], clp_df["color_val_norm"]])
    norm = TwoSlopeNorm(vmin=-np.nanmax(np.abs(color_all)), vcenter=0, vmax=np.nanmax(np.abs(color_all))) if center_zero else None

    figsize_x = max(6, len(common_mets) * base_width)
    fig, ax = plt.subplots(figsize=(figsize_x, figsize_y), dpi=dpi)
    sc_clp = ax.scatter(clp_x, clp_y, c=clp_df["color_val_norm"], s=clp_sizes, cmap=cmap, norm=norm, edgecolors="none", alpha=0.9)
    sc_lk  = ax.scatter(lk_x,  lk_y,  c=lk_df["color_val_norm"],  s=lk_sizes,  cmap=cmap, norm=norm, edgecolors="none", alpha=0.9)

    ax.set_xlim(-0.5, len(common_mets)-0.5)
    ax.set_ylim(-clp_bins.max()-0.5, lk_bins.max()+0.5)
    ax.axhline(0, color="gray", linestyle="--", lw=1, alpha=0.6)

    if sig_col not in sig_df.columns:
        sig_col = "pval_adj"
    xticks_labels = []
    for g in common_mets:
        p = sig_df.loc[g, sig_col] if g in sig_df.index else np.nan
        xticks_labels.append(f"{g} ({float(p):.2E})" if pd.notna(p) else g)
    ax.set_xticks(range(len(common_mets)))
    ax.set_xticklabels(xticks_labels, rotation=45, ha="right", fontsize=8)

    y_ticks = list(-np.sort(clp_bins)[::-1]) + [0] + list(np.sort(lk_bins))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([abs(int(v)) if v!=0 else "0" for v in y_ticks], fontsize=9)

    ax.set_xlabel("Metabolites", fontsize=11)
    ax.set_ylabel("Pseudotime (LK upward, CLP downward)", fontsize=11)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=1.2, length=4)

    # —— 关键改动：颜色条与“点大小图例”放在图右侧的独立小轴 —— 
    # 1) 放颜色条（靠近主图右侧）
    cbar = fig.colorbar(sc_lk, ax=ax, location="right", fraction=0.046, pad=0.02)
    cbar.set_label("Abundance (z-score across LK+CLP)", fontsize=10)
    cbar.ax.tick_params(labelsize=9, width=1)
    cbar.outline.set_linewidth(0.8); cbar.outline.set_edgecolor("gray")

    # 2) 计算主轴在图中的位置，基于它在右侧再开一块“大小图例轴”
    ax_pos = ax.get_position()  # [x0, y0, width, height] in figure fraction
    gap = 0.01                 # 主图/色条与大小图例的间隙
    legend_w = 0.10            # 大小图例轴的宽（figure fraction）
    legend_h = 0.28 * ax_pos.height  # 高度取主图高度的 28%
    legend_x = ax_pos.x1 + 0.07      # 放在颜色条的右边，适当留白
    legend_y = ax_pos.y0 + 0.10 * ax_pos.height  # 稍微居中偏下

    size_ax = fig.add_axes([legend_x, legend_y, legend_w, legend_h])
    size_ax.set_axis_off()

    # 在大小图例轴中画三个参考圆点
    # 轴范围设成 0~1，摆放纵向排列
    size_ax.set_xlim(0, 1); size_ax.set_ylim(0, 1)
    refs = [(1.00, 0.75, "100%"), (0.50, 0.50, "50%"), (0.25, 0.25, "25%")]
    for frac_ref, y, lab in refs:
        s = size_min + (size_max - size_min) * frac_ref
        size_ax.scatter([0.35], [y], s=s, facecolors="none", edgecolors="black", linewidths=0.9)
        size_ax.text(0.55, y, lab, va="center", fontsize=9)
    size_ax.text(0.5, 0.98, "Dot size = % expressing", ha="center", va="top", fontsize=9, color="gray")

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
    plt.show(); plt.close(fig)

