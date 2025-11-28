import os
import re
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import pyopenms as oms
from typing import Union
from pathlib import Path
from metacell.dataloader.scMetData import scMetData
from ._utils import get_index, get_tic_from_mzml, parse_filename


def load_rawdata(file: Union[str, Path], filename=None, marker_library=None, color_library=None) -> scMetData:
    """
    Load raw data from file.
    Include data with scan_Id, scan_start_time, TIC, mz, and intensity for all data points.

    Params:
    -------
    file: the path to raw scMet file.
    Returns:
    -------
    mdata: The updated object with added attributes: raw_scMet_data, mz_data, and intensity_data.
    """
    #  Retrieve the file extension and choose different file reading methods based on the specific extension.
    mdata = scMetData(file,filename)
    filetype = (os.path.splitext(file)[1])[1:]

    mdata.logger.info("Start the scMet data processing of {}.".format(mdata.filename))
    if filetype == "txt":
        mdata.raw_scm_data, mdata.mz_data, mdata.intensity_data = load_scMet_from_txt_file(file)

    elif filetype == "mzML":
        mdata.raw_scm_data, mdata.mz_data, mdata.intensity_data = load_scMet_from_mzML_file(file)
    else:
        sys.exit("File type not supported, it should be txt or mzML.")
    mdata.logger.info("Complete loading scMet raw data from {}.".format(mdata.filename))

    # 确保’TIC‘列中字符串转换为浮点数，无法转换的值将被设置为NaN
    mdata.raw_scm_data['TIC'] = pd.to_numeric(mdata.raw_scm_data['TIC'], errors='coerce')

    if color_library is not None:
        mdata.cell_type_color = color_library
    mdata.filename, mdata.cell_type_marker_df, mdata.cell_channel_df = parse_filename(mdata.filename, marker_library, color_library)

    return mdata


def load_scMet_from_txt_file(file: Union[str, Path]):
    """
    Load scMet information from a txt file.

    Params:
    -----------
    file: the path to raw scMet file.

    Returns:
    --------
    raw_scMet_data(df): Include data with scan_Id, scan_start_time, and TIC for all data points.
    mz_data(list): A list of data containing mz information for all data points.
    intensity_data(list): A list of data containing intensity information for all data points.
    """
    with open(file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    data = ''.join(lines)
    # === 1.extract scan_Id, scan_start_time, TIC ===
    # extract the scanId in the line where scanId=(.*).
    scan_Id = re.findall(r"id: scanId=(.*)", data)
    # match and retrieve the retention time scan_start_time in the line “cvParam: scan start time,(.*), minute” in the txt file.
    scan_start_time = re.findall(r"cvParam: scan start time, (.*), minute", data)
    scan_start_time = list(map(float, scan_start_time))
    # match and retrieve the TIC value in the line “cvParam: total ion current, (.*)” in the txt file.
    TIC = re.findall(r"cvParam: total ion current, (.*), number of detector counts", data)
    # Integrate scan_Id, scan_start_time and TIC.
    results = pd.DataFrame({
        'scan_Id': scan_Id,
        'scan_start_time': scan_start_time,
        'TIC': TIC
    })

    # === 2.extract all_mz all_intensity ===
    # extract the m/z and intensity-related information for each data point based on the cell_marker_mz values.
    # extract the index where the intensity is located.
    intensity_index = [i + 1 for i in get_index(lines, '          cvParam: intensity array, number of detector counts\n')]
    mz_index = [i + 1 for i in get_index(lines, '          cvParam: m/z array, m/z\n')]

    intensity_data = [
        np.array(lines[idx].strip().split()[2:], dtype=np.float64)
        for idx in tqdm(intensity_index, desc='All_intensity')
    ]

    mz_data = [
        np.array(lines[idx].strip().split()[2:], dtype=np.float64)
        for idx in tqdm(mz_index, desc='All_m_z')
    ]

    # The final data record in a file often contains chromatographic data,
    # resulting in a mismatch between the lengths of intensity_data and mz_data.
    if len(intensity_data) == len(mz_data) + 1:
        intensity_data = intensity_data[:-1]

    return results, mz_data, intensity_data


def load_scMet_from_mzML_file(file: Union[str, Path]):
    """
    Load scMet information from a mzML file.

    Params:
    -------
    file: the path to raw scMet file.

    Returns:
    --------
    raw_scMet_data(df): Include data with scan_Id, scan_start_time, and TIC for all data points.
    mz_data(list): A list of data containing mz information for all data points.
    intensity_data(list): A list of data containing intensity information for all data points.
    """
    # Prepare lists to store extracted data
    scan_Id, scan_start_time, TIC, mz_data, intensity_data = [], [], [], [], []
    # Create MSExperiment object and load mzML file
    exp = oms.MSExperiment()
    oms.MzMLFile().load(file, exp)

    # Iterate through all the spectra in the file
    for spectrum in exp.getSpectra():
        # Extract metadata
        scan_Id.append(int(spectrum.getNativeID().split('=')[1]))
        scan_start_time.append(spectrum.getRT() / 60)

        # Extract mzs and intensities
        mzs, intensities = spectrum.get_peaks()
        mz_data.append(mzs)
        intensity_data.append(intensities.astype('float64'))

        # Calculate TIC by summing all intensity values
        TIC.append(get_tic_from_mzml(spectrum))

    # Create DataFrame from extracted data
    results = pd.DataFrame({
        'scan_Id': scan_Id,
        'scan_start_time': scan_start_time,
        'TIC': TIC
    })

    results[['TIC', 'scan_start_time']] = results[['TIC', 'scan_start_time']].astype(float)
    return results, mz_data, intensity_data


# 示例
if __name__ == "__main__":
    pass
