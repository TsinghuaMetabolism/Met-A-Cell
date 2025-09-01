import os
import re
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from _utils import  calculate_signal_threshold_from_signal, calculate_signal_threshold_from_raw_data

# === Visualize the results of baseline correction. ===
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


def plt_baseline(signal_profile, output_dir=None):
    """
    Description:
    -----------
    Plot the original intensity and estimated baselines.


    Parameters:
    -----------
    signal_profile : pandas.DataFrame
        Must contain columns: 'time', 'data', 'baselines'.
    output_dir : str or None
        If provided, save figure to this directory.


    Returns:
    --------
    None
    """
    x = signal_profile['time']
    y1 = signal_profile['data']
    y2 = signal_profile['baselines']
    width = min(math.ceil(max(x)) * 6, 910)
    result_path = os.path.join(output_dir, 'plt_baselines.pdf') if output_dir else None


    plt.figure(figsize=(width, 6))
    plt.xlim(min(x), max(x) + 0.2)
    plt.axhline(0, color='white', linewidth=0.5)
    plt.plot(x, y1, color="black", linewidth=2.0, linestyle="solid", label="Raw intensity")
    plt.plot(x, y2, color="red", linewidth=2.0, linestyle="--", label="Baseline")
    plt.legend(loc='upper left')
    plt.xticks(np.arange(0, math.floor(max(x)) + 1, 1.0))


    if result_path:
        plt.savefig(result_path)
        plt.close()
    else:
        plt.show()

def plt_baseline_corrected(signal_profile, sn_ratio=3, output_dir=None):
    """
    Plot baseline-corrected signal with two threshold lines:
    - Orange: threshold from signal (baseline-corrected)
    - Green: threshold derived from raw data minus baseline


    Parameters:
    -----------
    signal_profile : pandas.DataFrame
        Must contain: 'time', 'signal', 'data', 'baselines'.
    sn_ratio : float
        Multiplier for standard deviation (used in both threshold methods).
    output_dir : str or None
        If provided, save the figure as PDF; otherwise show the plot.
    """
    x = signal_profile['time']
    y = signal_profile['signal']


    # Compute thresholds
    threshold_signal = calculate_signal_threshold_from_signal(signal_profile, multiplier=sn_ratio)
    #threshold_diff_array = calculate_signal_threshold_from_raw_data(signal_profile, multiplier=sn_ratio)


    # Set figure width
    width = min(math.ceil(max(x)) * 6, 910)
    result_path = os.path.join(output_dir, 'plt_baselines_correction.pdf') if output_dir else None


    # Create the figure
    plt.figure(figsize=(width, 6))
    plt.xlim(min(x), max(x) + 0.2)
    plt.plot(x, y, color="blue", linewidth=2.0, linestyle="solid", label="Corrected Signal")
    plt.plot(x, np.full(len(x), threshold_signal), color="orange", linewidth=2.0, linestyle="--", label=f"Threshold from Signal")
    #plt.plot(x, threshold_diff_array, color="green", linewidth=2.0, linestyle="--", label="Threshold from Raw Data")
    plt.xticks(np.arange(0, math.floor(max(x)) + 1, 1.0))
    plt.legend(loc='upper left')


    # Save or show the figure
    if result_path:
        plt.savefig(result_path)
        plt.close()
    else:
        plt.show()

def _plt_scm_events(x, data, scm_events_index, output_dir, figs_name ="plt_scm_events.pdf"):
    """
    Visualize the single-cell events annotated by marker.

    :param data: Original data.
    :param scm_events_index: Index of single-cell events.
    :param output_dir: path to output dir
    :param figs_name: figures name.
    :return: None
    """
    result_path = os.path.join(output_dir, figs_name)
    width = math.ceil(max(x)) * 6
    if width > 910:
        width = 910

    # 1) Plot the calibration of single-cell peaks annotated by TIC.
    plt.figure(figsize=(width, 6))
    plt.xlim(min(x), max(x) + 0.2)
    plt.plot(x, data, color="blue", linewidth=2.0, linestyle="solid", label="Raw data")
    plt.plot(x[scm_events_index], data[scm_events_index], "o", color="red", label="SCM events")
    plt.xticks(np.arange(0, math.ceil(max(x)) + 1, 1.0))  # Add x-axis tick marks.
    plt.savefig(result_path)
    plt.close()


def plt_scm_events(signal_profile, scm_events_index, output_dir=None, figs_name="plt_scm_events.pdf"):
    """
    Visualize the single-cell events on the original intensity trace.


    Parameters:
    -----------
    signal_profile : pandas.DataFrame
        Must contain 'time' and 'data' columns.
    scm_events_index : list or array-like
        Indices of detected single-cell events.
    result_path : str or None
        Full file path to save the figure. If None, the figure will be shown.
    figs_name : str
        File name for the saved figure (used only if result_path is not None).


    Returns:
    --------
    None
    """
    x = signal_profile['time']
    data = signal_profile['data']


    # Set figure width based on time range, limit max width
    width = min(math.ceil(max(x)) * 6, 910)


    # Create figure
    plt.figure(figsize=(width, 6))
    plt.xlim(min(x), max(x) + 0.2)
    plt.plot(x, data, color="blue", linewidth=2.0, linestyle="solid", label="Raw data")
    plt.plot(x.iloc[scm_events_index], data.iloc[scm_events_index], "o", color="red", label="SCM events")
    plt.xticks(np.arange(0, math.ceil(max(x)) + 1, 1.0))
    plt.legend(loc='upper left')


    # Save or show
    if output_dir:
        result_path = os.path.join(output_dir, figs_name)
        plt.savefig(result_path)
        plt.close()
    else:
        plt.show()


def _plt_merged_scm(x, data, scm_events_index, scm_events_only_index, output_dir):
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
    # Define the output path for the plot
    result_path = os.path.join(output_dir, 'plt_merged_scMetEvent.pdf')
    # Calculate the figure width based on the data range, with a maximum width limit
    custom_colors = ['#FFC125', '#9ACD32', '#7B68EE', '#EE7942', '#6CA6CD']
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

    # set x-axis ticks
    plt.xticks(np.arange(0, math.ceil(max(x)) + 1, 1.0))
    # Add legend to the plot
    plt.legend()
    # Save the plot to the specified output directory
    plt.savefig(result_path)
    plt.close()

def plt_merged_scm(signal_profile, scm_events_index, scm_events_only_index, output_dir=None):
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


    # set x-axis ticks
    plt.xticks(np.arange(0, math.ceil(max(x)) + 1, 1.0))
    # Add legend to the plot
    plt.legend(loc='upper left')


    # Save the plot to the specified output directory
    if output_dir:
      result_path = os.path.join(output_dir, 'plt_merged_scMetEvent.pdf')
      plt.savefig(result_path)
      plt.close()
    else:
      plt.show()


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
                              annotated_index=cell_type_marker_scm[f'{marker_name}_index'], color=color)
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

def plt_annotated_subplot(axs, x, intensity, annotated_index, color):
    """
    绘制带有标注的子图。
    """
    axs.plot(x[annotated_index], intensity[annotated_index],
             linestyle='none', marker="*", markersize='10', color=color)

def plt_annotated_scatter(axs, time, intensity, color):
    """
    绘制带有散点标注的图像。
    """
    axs.scatter(time, intensity, marker="*",color=color,s=100,alpha=1,zorder=3)