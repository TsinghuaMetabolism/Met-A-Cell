import os
import math
import numpy as np
import matplotlib.pyplot as plt

# === Visualize the results of baseline correction. ===

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
    from ._utils import calculate_signal_threshold_from_signal, calculate_signal_threshold_from_raw_data
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

def _plt_scm_events(x, data, scm_events_index, output_dir, figs_name ="plt_scMetEvent.pdf"):
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

