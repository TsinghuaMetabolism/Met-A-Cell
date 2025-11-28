import os
import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plt_tic_time(raw_scm_data, show=True, save_path=None, dpi=300, width=10):
    """
    Visualize the Total Ion Current (TIC) trajectory over scan start time
    from raw single-cell metabolomics data.


    Parameters:
    - raw_scm_data: pd.DataFrame
        DataFrame containing 'scan_start_time' and 'TIC' columns.
    - show: bool, default True
        Whether to display the figure.
    - save_path: str or None, default None
        File path to save the figure as PDF (e.g., 'tic_trajectory.pdf').
    - dpi: int, default 300
        Resolution for saved figure (ignored for vector formats like PDF).
    - width: int or float, default 10
        Width of the output figure (height is fixed at 5 inches).
    """
    if 'scan_start_time' not in raw_scm_data.columns or 'TIC' not in raw_scm_data.columns:
        raise ValueError("Input data must contain 'scan_start_time' and 'TIC' columns.")

    # Set clean style without background grid
    sns.set_style("white")

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(width, 5))

    # Plot TIC curve
    ax.plot(raw_scm_data['scan_start_time'], raw_scm_data['TIC'],
            color='steelblue', linewidth=1.5)

    # Set labels and title with styling
    ax.set_xlabel("Scan Start Time", fontsize=13, fontweight='bold', color='black')
    ax.set_ylabel("Total Ion Current (TIC)", fontsize=13, fontweight='bold', color='black')
    ax.set_title("TIC Profile Across Scan Time", fontsize=15, fontweight='bold', color='black')

    # Set tick label font and color
    ax.tick_params(axis='both', which='major',
                   labelsize=11, labelcolor='#333333',
                   color='#666666',
                   length=6, width=1, direction='out')

    # Set x-axis start at 0
    ax.set_xlim(left=0)

    # Remove top/right border
    sns.despine()

    # Adjust layout
    plt.tight_layout()

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=dpi, format='pdf')
        print(f"Figure saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plt_scm_events(signal_profile, scm_events_index, output_dir=None, width=None, figs_name="plt_scMetEvent.pdf"):
    """
    Visualize the single-cell events on the original intensity trace.

    Parameters
    ----------
    signal_profile : pandas.DataFrame
        Must contain 'time' and 'data' columns.
    scm_events_index : list or array-like
        Indices of detected single-cell events.
    output_dir : str or None
        Directory to save the figure. If None, the figure will be shown.
    figs_name : str
        File name for the saved figure (used only if output_dir is not None).

    Returns
    -------
    None
    """
    import os, math
    import numpy as np
    import matplotlib.pyplot as plt

    x = signal_profile['time']
    data = signal_profile['data']

    # Set figure width based on time range, limit max width
    if width is None:
        width = min(math.ceil(max(x)) * 6, 910)

    # Create figure
    plt.figure(figsize=(width, 6))
    plt.xlim(min(x), max(x) + 0.2)
    plt.plot(x, data, color="blue", linewidth=2.0, linestyle="solid", label="Raw data")
    plt.plot(x.iloc[scm_events_index], data.iloc[scm_events_index], "o", color="red", label="SCM events")

    # --- 给红点添加编号 ---
    for i, idx in enumerate(scm_events_index, start=1):
        plt.annotate(
            str(i),
            (x.iloc[idx], data.iloc[idx]),
            xytext=(5, 5),  # 偏移，避免和点重叠
            textcoords="offset points",
            fontsize=8,
            color="black"
        )

    # Set x-axis ticks
    plt.xticks(np.arange(0, math.ceil(max(x)) + 1, 1.0))
    plt.legend(loc='upper left')

    # Save or show
    if output_dir:
        result_path = os.path.join(output_dir, figs_name)
        plt.savefig(result_path)
        plt.close()
    else:
        plt.show()


def plt_merged_scm(signal_profile, scm_events_index, scm_events_only_index, output_dir=None, width=None, figs_name="plt_merged_scMetEvent.pdf"):
    """
    Description:
    -----------
    Visualize the single-cell peaks annotated by merged scMetEvent.


    Parameter:
    ----------
    data(df)
    merged_scMetEvent_index(array)
    scMetEvent_only_index(dict) wwwwwww
    output_dir(str)


    Returns
    -------
    None
    """
    x = signal_profile['time']
    data = signal_profile['data']
    # Calculate the figure width based on the data range, with a maximum width limit
    custom_colors = ['#FFC125', '#9ACD32', '#7B68EE', '#EE7942', '#6CA6CD']
    if width is None:
        width = min(math.ceil(max(x)) * 6, 910)


    # Create the figure with calculated dimensions
    plt.figure(figsize=(width, 6))
    plt.xlim(min(x), max(x) + 0.2)


    # Plot the raw TIC data
    plt.plot(x, data, color="blue", linewidth=2.0, linestyle="solid", label="Raw TIC data")


    # Plot single-cell events annotated by multiple strategies.
    plt.plot(x[scm_events_index], data[scm_events_index], "o", color="#B22222", label="scMetEvent annotated by multiple strategies.")


    for i, (key, value) in enumerate(scm_events_only_index.items()):
        # Plot single-cell events annotated by cell marker only
        plt.plot(x[value], data[value], "o", color=custom_colors[i], label=f"scMetEvent annotated by {key} only")


    for i, idx in enumerate(scm_events_index, start=1):
        plt.annotate(
            str(i),
            (x.iloc[idx], data.iloc[idx]),
            xytext=(5, 5),  # 偏移，避免和点重叠
            textcoords="offset points",
            fontsize=8,
            color="black"
        )

    # set x-axis ticks
    plt.xticks(np.arange(0, math.ceil(max(x)) + 1, 1.0))
    # Add legend to the plot
    plt.legend(loc='upper left')


    # Save the plot to the specified output directory
    if output_dir:
      result_path = os.path.join(output_dir, figs_name)
      plt.savefig(result_path)
      plt.close()
    else:
      plt.show()


def plt_scMet(mdata, type, output_dir=None, width=None):
    """
    Unified plotting function that integrates plt_scm_events and plt_merged_scm.

    Parameters:
    - mdata: An object containing attributes such as cell_marker_eic, scm_events_index, and scm_events_only_index.
    - type: The plotting type. Should be one of the keys in mdata.scm_events_index or in the format 'merged_{marker}'.
    - output_dir: Directory to save the output plots (optional).
    - width: Width of the plot (optional).
    - figs_name: File name to save the figure when using plt_scm_events.
    """
    # Get available types from scm_events_index
    available_types = list(mdata.scm_events_index.keys())

    # Check if the specified type is valid
    if type not in available_types:
        raise ValueError(f"Invalid 'type': {type}. It should be one of {available_types}")

    if type == 'merged':
        # Plot using the main marker for merged type
        signal_profile = mdata.cell_marker_eic[mdata.main_cell_marker]
        plt_merged_scm(signal_profile=signal_profile,
                       scm_events_index=mdata.scm_events_index['merged'],
                       scm_events_only_index=mdata.scm_events_only_index,
                       output_dir=output_dir,
                       width=width,
                       figs_name="plt_merged_scMetEvent.pdf")

    elif type.startswith('merged_'):
        # Plot merged signal for a specific marker
        marker = type.replace('merged_', '')
        if marker not in mdata.cell_marker_eic:
            raise ValueError(f"Invalid marker '{marker}' in 'merged_{marker}'. It should be one of {list(mdata.cell_marker_eic.keys())}")
        signal_profile = mdata.cell_marker_eic[marker]
        plt_merged_scm(signal_profile=signal_profile,
                       scm_events_index=mdata.scm_events_index[type],
                       scm_events_only_index=mdata.scm_events_only_index,
                       output_dir=output_dir,
                       width=width,
                       figs_name=f"plt_merged_{marker}_scMetEvent.pdf")

    elif type in mdata.cell_marker_eic:
        # Plot signal for a specific marker using standard SCM event plotting
        signal_profile = mdata.cell_marker_eic[type]
        plt_scm_events(signal_profile,
                       scm_events_index=mdata.scm_events_index[type],
                       output_dir=output_dir,
                       width=width,
                       figs_name=f"plt_scMetEvent_by_{type}.pdf")

def plt_feature_eic(mdata, mz, output_dir=None, fig_name="feature_plot",
                    interval=6, offset: int = 1,
                    sn_ratio: float = 3, ppm_threshold: float = 10,
                    metab_anno: pd.DataFrame = None):
    """
    Plot two subplots: SCM events on the cell marker signal and the EIC signal of a given m/z.

    Parameters
    ----------
    mdata : object
        The object containing single-cell metabolomics data.
    mz : float
        Target m/z value to extract EIC.
    output_dir : str or None
        Output directory for saving figures. If None, the figure will be displayed instead of saved.
    fig_name : str
        Base name for the figure. The actual filename will be "{fig_name}_mz={mz}.pdf".
    interval : int
        Minimum distance between detected peaks.
    offset : int
        Offset range to determine overlaps between peaks and SCM events.
    sn_ratio : float
        Signal-to-noise ratio threshold for peak detection.
    ppm_threshold : float
        PPM tolerance for extracting EIC.
    metab_anno : pd.DataFrame or None
        Optional annotation DataFrame containing columns ["metabolite", "mz"].

    Returns
    -------
    None
    """
    from pathlib import Path
    from ..dataloader._utils import sliding_window_baseline_correction, detect_peaks, calculate_signal_threshold_from_signal
    from ..dataloader._utils import double_scan_feature_integration, calculate_intersection
    # Define custom colors
    color = {
        "gray": "#E2E1E0", "purple": "#bd4da3", "blue": "#88c6e2",
        "darkpurple": "#8182BA", "red": "#C2989C",
        "green": "#478084", "orange": "#fbb01a"
    }

    # -------- Subplot 1: Cell marker signal and SCM events --------
    signal_profile = mdata.cell_marker_eic[mdata.main_cell_marker]
    scm_events_index = mdata.scm_events.index
    x = signal_profile['time']
    data = signal_profile['data']

    fig_width = min(math.ceil(max(x)) * 6, 910)
    fig, axes = plt.subplots(2, 1, figsize=(fig_width, 10), sharex=False)

    # -------- EIC extraction and processing --------
    mz_data = mdata.mz_data
    intensity_data = mdata.intensity_data

    density_center, mz_intensity = double_scan_feature_integration(
        mz_data, intensity_data, mz, ppm_threshold
    )

    df = sliding_window_baseline_correction(
        mz_intensity['intensity'], x,
        sn_ratio, output2figures=None, window_size=100, p=0.5
    )

    mph = calculate_signal_threshold_from_signal(df, multiplier=sn_ratio)
    peaks = detect_peaks(df['signal'], mpd=interval, mph=mph)

    intersect1_e2, intersect2_e1, _, _ = calculate_intersection(scm_events_index, peaks, offset)

    # -------- Plot subplot 1 --------
    axes[0].plot(x, data, color=color["blue"], linewidth=2.0, linestyle="solid", label="Cell marker signal")
    axes[0].plot(x.iloc[scm_events_index], data.iloc[scm_events_index], "o", color=color["purple"], label="scMet events")
    axes[0].plot(x.iloc[intersect1_e2], data.iloc[intersect1_e2], "*", color=color["orange"], markersize=10, label="scMet ∩ EIC peaks")
    axes[0].set_xlim(min(x), max(x) + 0.2)
    axes[0].set_xticks(np.arange(0, math.ceil(max(x)) + 1, 1.0))
    axes[0].set_ylabel("Intensity")
    axes[0].legend(loc='upper left')

    # -------- Plot subplot 2 --------
    axes[1].plot(x, df['signal'], color=color["blue"], linewidth=2.0, label="EIC signal")
    axes[1].plot(x.iloc[intersect2_e1], df['signal'].iloc[intersect2_e1], "o", color=color["orange"], label="scMet ∩ EIC peaks")
    axes[1].set_xlim(min(x), max(x) + 0.2)
    axes[1].set_xticks(np.arange(0, math.ceil(max(x)) + 1, 1.0))
    axes[1].set_ylabel("Intensity")
    axes[1].set_xlabel("Time")
    axes[1].legend(loc='upper left')

    # -------- Generate title --------
    title = f"Feature m/z = {mz:.4f}"

    # If annotation is provided, add metabolite names to title
    if metab_anno is not None and "mz" in metab_anno.columns and "metabolite" in metab_anno.columns:
        matched = metab_anno[abs(metab_anno["mz"] - mz) / mz * 1e6 <= 10]
        if not matched.empty:
            metab_name_str = "/".join(matched["metabolite"].unique())
            title += f" ({metab_name_str})"

    # -------- Set left-aligned figure title --------
    fig.suptitle(title, x=0.0, ha='left', fontsize=14, fontweight='bold')

    # -------- Show or save the figure --------
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if output_dir is None:
        plt.show()
    else:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_file = os.path.join(output_dir, f"{fig_name}_mz={mz}.pdf")
        plt.savefig(output_file)
        plt.close()

def plt_cell_type_marker_annotation(mdata, cell_type_marker, result_path: str=None, offset=1):
    # cell_type_marker = mdata.cell_type_marker_df
    main_cell_marker = mdata.main_cell_marker
    x = mdata.raw_scm_data['scan_start_time']
    width = min(math.ceil(max(x)) * 6, 910)
    if result_path is not None:
        figs_output_dir = os.path.join(result_path, 'Figures/cell_type_marker/')
        os.makedirs(figs_output_dir, exist_ok=True)
    else:
        figs_output_dir = None

    if len(mdata.cell_marker) == 1:
        scm_events_index = mdata.scm_events_index[main_cell_marker]
    else:
        scm_events_index = mdata.scm_events_index['merged']

    i = 0
    plt_number = len(cell_type_marker)+1
    fig, axs = plt.subplots(plt_number, 1, figsize=(width, 6 * plt_number))
    # 绘制 cell marker
    main_cell_marker_intensity = 'TIC' if mdata.main_cell_marker == 'TIC' else f'{mdata.main_cell_marker}_intensity'

    plt_peak_subplot(axs=axs[i], x=x, intensity=mdata.raw_scm_data[main_cell_marker_intensity],
                      peak_index=mdata.scm_events_index[main_cell_marker], line_color='blue',
                      o_color='#D3D3D3', title=main_cell_marker)
    plt_annotated_scatter(axs[i], mdata.scm_events['scan_start_time'], mdata.scm_events[main_cell_marker_intensity],'#D3D3D3')
    axs[i].scatter(mdata.scm_events['scan_start_time'], mdata.scm_events[main_cell_marker_intensity],
                    marker="*", color='#D3D3D3',s=100,alpha=1)

    plt_add_cell_number(axs[i], mdata.scm_events['scan_start_time'], mdata.scm_events[main_cell_marker_intensity], mdata.scm_events['CellNumber'], 'number')

    i = i + 1

    for _, row in cell_type_marker.iterrows():
        marker_name = row['marker_name']
        color = row['color_code']
        mz = row['mz']
        eic = mdata.cell_type_marker_eic[marker_name]
        cell_type_marker_scm = mdata.scm_events[mdata.scm_events[f'{marker_name}_index'].notna()]

        plt_peak_subplot(axs[i], x=x, intensity= eic['data'],
                          peak_index= mdata.cell_type_marker_apex_index[marker_name],
                          line_color='blue', o_color='#D3D3D3', title=f'{marker_name}: {mz}')
        plt_annotated_subplot(axs=axs[i], x=x, intensity=eic['data'],
                              annotated_index=cell_type_marker_scm[f'{marker_name}_index'], color=color,marker_name=marker_name)
        cell_type_markr_apex = eic.loc[mdata.cell_type_marker_apex_index[marker_name]].loc[cell_type_marker_scm[f'{marker_name}_index']]

        plt_add_cell_number(axs[i], cell_type_marker_scm['scan_start_time'], cell_type_markr_apex['data'], cell_type_marker_scm['CellNumber'],'number')
        i = i + 1

    plt_annotated_scatter(axs[0], mdata.scm_events['scan_start_time'], mdata.scm_events[main_cell_marker_intensity],
                          mdata.scm_events['cell_type_encode'].apply(mdata.cell_type_marker_encoder.encode_to_color))

    if figs_output_dir is not None:
       plt.savefig(f'{figs_output_dir}/cell_type_marker_annotation.pdf')
       plt.close()

def plt_peak_subplot(axs, x, intensity, peak_index, line_color, o_color, title):
    """
    绘制峰谱子图并标注峰值点。
    """
    axs.set_xlim(min(x), max(x) + 0.2)
    axs.plot(x, intensity, color=line_color, linewidth=2.0, linestyle="solid", label="Raw data")
    axs.plot(x[peak_index], intensity[peak_index], linestyle='none',
              marker="o", color=o_color,label=f'marker: {title}',zorder=2)
    axs.set_title(title, loc='left', fontsize=20, fontstyle='italic', pad=10)
    axs.set_xticks(np.arange(0, math.ceil(max(x)) + 1, 1.0))
    axs.legend(loc='upper left')

def plt_add_cell_number(axs, time, intensity, cellnumber, cellnumber_type='text'):
    """
    在图中标注细胞编号
    """
    if cellnumber_type == 'number':
        # 提取数字部分并将其转换为整数
        cellnumber = [int(re.search(r'\d+', cn).group()) for cn in cellnumber]

    for cell_number, x, y in zip(cellnumber, time, intensity):
        axs.annotate(cell_number, (x, y),
                     textcoords="offset points",
                     xytext=(0, 10),
                     ha='center',
                     rotation=45,  # 字体旋转45度
                     fontsize=20 if cellnumber_type == 'number' else 10,
                     arrowprops=dict(facecolor='black', arrowstyle="->"))

def plt_annotated_subplot(axs, x, intensity, annotated_index, color,marker_name):
    """
    绘制带有标注的子图。
    """
    axs.plot(x[annotated_index], intensity[annotated_index],
             linestyle='none', marker="*", markersize='10', color=color,label=marker_name)
    axs.legend(loc='upper left')

def plt_annotated_scatter(axs, time, intensity, color):
    """
    绘制带有散点标注的图像。
    """
    axs.scatter(time, intensity, marker="*",color=color,s=100,alpha=1,zorder=3)
    axs.legend(loc='upper left')

