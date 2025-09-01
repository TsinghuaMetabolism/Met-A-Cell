import re
import os
import pandas as pd
import numpy as np
from ._utils import  sliding_window_baseline_correction, detect_peaks
from ._utils import calculate_signal_threshold_from_signal

def load_lif_data(mdata, input_path, filename, sn_ratio=3, interval=6):
    """
    load lif data
    """
    all_files = find_files_with_prefix_and_suffix(input_path, filename, ['csv', 'CSV'])
    mdata.lif_data_dir = {}
    mdata.lif_peak_data_dir = {}

    if read_lif_data(mdata, all_files):
        for filename, lif_data in mdata.lif_data_dir.items():
            mdata.logger.info(f'Start extracting lif peak data from {filename}')
            mdata.logger.info("-" * 50)
            mdata.lif_peak_data_dir[filename] = extract_lif_peak(mdata, lif_data, interval, sn_ratio)
            mdata.logger.info("-" * 50)

def read_lif_data(mdata, all_files):
    """
    Check if the lif file exists (ignoring case for suffix), read the file, and return True.
    Return False if the file does not exist.
    Returns:
    --------
    bool
        True if the lif file is read successfully, False otherwise.
    """
    pattern = re.compile(r'_(.*?)\.[^.]+$')

    lif_data_dir = {}

    if all_files:
        for file_path in all_files:
            file = os.path.basename(file_path)
            filename = pattern.search(file).group(1)
            lif_data_dir[filename] = pd.read_csv(file_path, encoding="utf-16", sep='\t', header=None,
                                                      names=['times', 'intensity'])
            print(f"lif data {file} has been read successfully.")
    mdata.lif_data_dir = lif_data_dir
    if lif_data_dir:
        return True
    else:
        print(f"No lif file found.")
        return False

def find_files_with_prefix_and_suffix(path, prefix, suffixes):
    # Create a regex pattern to match the files
    pattern = re.compile(rf'^{re.escape(prefix)}.*\.({"|".join(suffixes)})$', re.IGNORECASE)

    matching_files = []

    for root, _, files in os.walk(path):
        for file in files:
            if pattern.match(file):
                matching_files.append(os.path.join(root, file))

    return matching_files

def extract_lif_peak(mdata, lif_data, interval=60, sn_ratio=3, window_size=1000):
    """
    Extract the peak spectral information from the lif data.

    Parameters:
    -----------
    window_size : int, optional
        The size of the sliding window for baseline correction (default is 1000).

    Returns:
    --------
    None
    """
    # Perform sliding window baseline correction
    mdata.logger.info(f'1) Perform sliding window baseline correction')
    df = sliding_window_baseline_correction(lif_data['intensity'], lif_data['times'], sn_ratio,
                                                    output2figures=None, window_size=window_size, p=0.5)
    # Detect peaks in the intensity data
    mdata.logger.info(f'2) Detect peaks in the intensity data')
    mph = calculate_signal_threshold_from_signal(df, multiplier=sn_ratio)
    peaks = detect_peaks(lif_data['intensity'], mpd=interval, mph=mph)

    # Check and filter the detected peaks events
    mdata.logger.info(f'3) Check and filter the detected peaks events')

    checked_lif_peak = check_lif_peak(df['baselines'], peaks, test_range=1000, threshold=0.1)

    # Update instance attributes with the detected peaks
    lif_peak_data = lif_data.copy()
    lif_peak_data['baselines'] = df['baselines']
    lif_peak_data = lif_peak_data.iloc[checked_lif_peak]
    lif_peak_data.loc[:, 'corrected_intensity'] = np.where(
        lif_peak_data['intensity'] > lif_peak_data['baselines'],
        lif_peak_data['intensity'] - lif_peak_data['baselines'],
        0
    )

    return lif_peak_data

def check_lif_peak(baselines, index, test_range=1000, threshold=0.2):
    # Create an empty array to store the index values where the difference of means is greater than the threshold.
    significant_diff = []

    # Iterate through each peak point.
    for i in index:
        # Ensure that the starting index of the left-side data is not less than 0.
        left_start = max(0, i - test_range)
        # Ensure that the ending index of the right-side data is not greater than the length of the baselines.
        right_end = min(len(baselines), i + test_range)

        # Extract the data from both the left and right sides.
        left_data = baselines[left_start:i]
        right_data = baselines[i:right_end]

        # Calculate the mean of the data on both sides.
        mean_diff = np.mean(right_data) - np.mean(left_data)

        # If the difference between the means is greater than the threshold, record the index.
        if abs(mean_diff) < threshold:
            significant_diff.append(i)

    return np.array(significant_diff)
