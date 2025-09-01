import os
import math
import numpy as np
import matplotlib.pyplot as plt

def plt_peaks(time, data, peaks, output_dir=None, figs_name="plt_scMetEvent.pdf"):
    """
    Description
    ------------
    Visualize the single-cell peaks annotated by TIC.

    Parameters
    ----------
    time : pandas.DataFrame
    data: Original data.
    peaks: Index of single-cell peaks.
    output_dir: path to output dir
    figs_name: str

    Returns
    -------
    None
    """
    width = math.ceil(max(time)) * 6
    if width > 910:
        width = 910

    # 1) Plot the calibration of single-cell peaks annotated by TIC.
    plt.figure(figsize=(width, 6))
    plt.xlim(min(time), max(time)+0.2)
    plt.plot(time, data, color="blue", linewidth=2.0, linestyle="solid", label="Raw data")
    plt.plot(time[peaks], data[peaks], "o", color="red", label="scMetEvent")
    plt.xticks(np.arange(0, math.ceil(max(time)) + 1, 1.0))  # Add x-axis tick marks.
    if output_dir is not None:
        result_path = os.path.join(output_dir, figs_name)
        plt.savefig(result_path)
        plt.close()
    else:
        plt.show()
