import os
import numpy as np
import pandas as pd

from metacell.dataloader.scMetData import scMetData
from ._utils import double_scan_feature_integration, sliding_window_baseline_correction, detect_peaks
from ._utils import calculate_signal_threshold_from_signal
from ._plotting import _plt_scm_events, _plt_merged_scm


def extract_scm_events(mdata: scMetData, cell_marker: dict, main_cell_marker: str = None, result_path: str = None,
                       poor_signal_range: pd.DataFrame = None, offset: int = 1, sn_ratio: int = 3,
                       interval: int = 6, ppm=10):
    """
    Extract SCM events from raw data based on the given cell markers.

    :param mdata: The scMetData object containing raw data.
    :param cell_marker: A dictionary recording features used as cell markers.
    :param main_cell_marker: The main feature used as the cell marker.
    :param result_path:  path to output figures.
    :param poor_signal_range: Time periods to discard. 每一行记录一段需要丢弃的时间段。start和end分别记录时间段的起始和
    :param offset
    :param sn_ratio: Signal-to-noise ratio threshold.
    :param interval: Minimum interval between cell events.
    :return:
    """
    # Check if cell_marker is empty
    if not cell_marker:
        # Log a warning and set default values.
        mdata.logger.warning("cell_marker is empty. Defaulting to {'TIC': None} and setting main_cell_marker to 'TIC'")
        cell_marker = {'TIC': None}
        main_cell_marker = 'TIC'

    else:
        # If cell_marker has only one key
        if len(cell_marker) == 1:
            # Set main_cell_marker to that key
            main_cell_marker = list(cell_marker.keys())[0]
            mdata.scm_type = main_cell_marker
        else:
            # If cell_marker has multiple keys
            if main_cell_marker not in cell_marker.keys():
                # Log a warning and check for 'TIC'
                if 'TIC' in cell_marker.keys():
                    mdata.logger.warning(
                        f"main_cell_marker '{main_cell_marker}' is not in cell_marker. Setting main_cell_marker to 'TIC'")
                    main_cell_marker = 'TIC'
                else:
                    # Log an error and raise an exception
                    mdata.logger.error(f"main_cell_marker '{main_cell_marker}' is not in cell_marker, "
                                       f"and 'TIC' is also not present in cell_marker. Please provide a valid main_cell_marker.")
                    raise ValueError(
                        f"main_cell_marker '{main_cell_marker}' is not in cell_marker, "
                        f"and 'TIC' is also not present in cell_marker. Please provide a valid main_cell_marker."
                    )
            mdata.scm_type = 'merged'
    mdata.cell_marker = cell_marker
    mdata.main_cell_marker = main_cell_marker
    # Filter data according to the given time range, setting the corresponding intensity values to NaN.
    if poor_signal_range is not None:
        mdata.raw_scm_data = discard_time_period(mdata.raw_scm_data, time_period=poor_signal_range)

    for marker in cell_marker.keys():
        mdata.logger.info(f"Start extracting SCM events by {marker}.")

        if result_path is None:
            figs_output_dir = None
        else:
            figs_output_dir = os.path.join(result_path, 'Figures/scMetEvent/', marker, '')
            os.makedirs(figs_output_dir, exist_ok=True)

        if marker != 'TIC':
            mz_intensity = extract_intensity(mdata, name=marker, mz=cell_marker[marker],ppm=ppm)
            mdata.raw_scm_data.update(mz_intensity)
            mdata.raw_scm_data = pd.concat([mdata.raw_scm_data, mz_intensity.loc[:, ~mz_intensity.columns.isin(mdata.raw_scm_data.columns)]], axis=1)

        # 需要加一个判断，如果mdata.scm_events_index的key不为空，且key中包含有marker，则跳过。
        mdata.cell_marker_eic[marker], mdata.scm_events_index[marker] = extract_peak(mdata, marker=marker,
                                                                                     figs_output_dir=figs_output_dir,
                                                                                     sn_ratio=sn_ratio,
                                                                                     interval=interval)

        mdata.logger.info(f"Finish extracting SCM events by {marker}.")

    if 'TIC' not in cell_marker.keys():
        marker = 'TIC'
        mdata.cell_marker_eic[marker], _ = extract_peak(mdata, marker=marker,
                                                        figs_output_dir=None,
                                                        sn_ratio=sn_ratio,
                                                        interval=interval)

    if len(cell_marker) == 1:
        mdata.scm_events = mdata.raw_scm_data.iloc[mdata.scm_events_index[main_cell_marker]]
        cellnumber = ['Cell{:05d}'.format(i + 1) for i in range(len(mdata.scm_events))]
        mdata.scm_events.loc[:, 'CellNumber'] = cellnumber

    else:
        mdata.logger.info(f'Start the integration of the results of multiple strategies')
        mdata = merge_scm_events(mdata, cell_marker, main_cell_marker, result_path, offset)
        mdata.logger.info(f'Complete the integration of the results of multiple strategies')

    # 解析cell_marker字典，确定提取策略。
    if len(cell_marker) == 1:
        strategy = list(cell_marker.keys())[0]
    else:
        keys = [main_cell_marker] + [k for k in cell_marker.keys() if k != main_cell_marker]
        strategy = ', '.join(keys)
    mdata.processing_status['scm_events_extraction_strategy'] = strategy

    return mdata

def extract_intensity(mdata: scMetData, name: str, mz: float, ppm=10) -> pd.DataFrame:
    """
    Extract the mz and its corresponding intensity from all data points that meet the conditions.

    Returns:
    --------
    mz_intensity:
    """
    # Perform double scan feature integration to get density center and intensity data
    density_center, mz_intensity = double_scan_feature_integration(mdata.mz_data, mdata.intensity_data, mz, ppm=ppm)
    # Rename the columns of the resulting DataFrame
    mz_intensity.columns = [f"{name}_{col}" for col in mz_intensity.columns]
    # Apply median filtering to fill missing values in the signal.
    # Fill missing intensity values with the median intensity
    #mz_intensity[f"{name}_intensity"].fillna(mz_intensity[f"{name}_intensity"].median(), inplace=True)

    mdata.logger.info(f"Complete 2-step scanning to obtain the mz and intensity from marker {name} : {density_center}.")
    return mz_intensity


def extract_peak(mdata: scMetData,
                 marker: str,
                 figs_output_dir: str,
                 sn_ratio: int=3,
                 interval: int=6):
    """
    从数据中提取信号峰。

    Params:
    -------

    Returns:
    -------
    """
    if marker == 'TIC':
        marker_intensity = marker
    else:
        marker_intensity = f'{marker}_intensity'

    # baseline correction by step and confirmation of valid signal index based on S/N threshold
    df = sliding_window_baseline_correction(mdata.raw_scm_data[marker_intensity],
                                                        mdata.raw_scm_data['scan_start_time'], sn_ratio,
                                                        figs_output_dir,
                                                        window_size=100, p=0.5)
    mph = calculate_signal_threshold_from_signal(df, multiplier=sn_ratio)

    # Use detect_peaks to find all peak points in the data and obtain their indices.
    peaks = detect_peaks(df['signal'], mpd=interval, mph=mph)

    if figs_output_dir is not None:
        mdata.logger.info(f'Complete visualization of {marker}-annotated single-cell events: plt_baselines.pdf and plt_baselines_correction.pdf')
        _plt_scm_events(mdata.raw_scm_data['scan_start_time'], mdata.raw_scm_data[marker_intensity],
                       peaks, figs_output_dir)

        mdata.logger.info(f'Complete visualization of {marker}-annotated single-cell events: plt_scm_event.pdf.')

    return df, peaks

def discard_time_period(df: pd.DataFrame, time_period: pd.DataFrame):
    """
    Filter data according to the given time range, setting the corresponding intensity values to NaN.

    Params:
    -------
    time_period: 将要去除的时间段。

    Returns:
    -------
    df: Modified datasets with the ‘TIC’ column values of data points within the periods of poor signal quality set to NaN.
    """
    # Select columns ending with ‘intensity’ and add 'TIC' to the list
    intensity_columns = df.columns[df.columns.str.endswith('intensity')].tolist()
    intensity_columns.append('TIC')

    # Iterate over each row in time period to mask the data.
    for _, row in time_period.iterrows():
        mask = (df['scan_start_time'] >= row['start']) & (df['scan_start_time'] <= row['end'])
        df.loc[mask, intensity_columns] = np.nan

    return df


def merge_scm_events(mdata: scMetData, cell_marker: dict, main_cell_marker: str, result_path: str = None,
                     offset: int = 1):
    """
    Integrate the results of multiple strategies to obtain merged scMetEvent.

    :param mdata:
    :param cell_marker:
    :param main_cell_marker:
    :param result_path:
    :param offset:
    :return:
    """
    # Robustly integrate single-cell peak annotations from multiple strategies according to the offset
    mdata.scm_events_index['merged'], mdata.scm_events_only_index = robust_scm_events_integration(mdata, cell_marker,
                                                                                            main_cell_marker, offset)

    if result_path is not None:
        figs_output_dir = os.path.join(result_path, 'Figures/scMetEvent/merged/')
        os.makedirs(figs_output_dir, exist_ok=True)

        cell_marker_intensity = main_cell_marker if main_cell_marker == 'TIC' else f'{main_cell_marker}_intensity'
        _plt_merged_scm(mdata.raw_scm_data['scan_start_time'], mdata.raw_scm_data[cell_marker_intensity],
                       mdata.scm_events_index['merged'], mdata.scm_events_only_index, figs_output_dir)

        mdata.logger.info(f'Complete visualization of merged SCM events: plt_merged_scm_event.pdf')

    mdata.scm_events = mdata.raw_scm_data.iloc[mdata.scm_events_index['merged']]
    cellnumber = ['Cell{:05d}'.format(i + 1) for i in range(len(mdata.scm_events))]
    mdata.scm_events.loc[:, 'CellNumber'] = cellnumber

    mdata.logger.info(f'(1) The number of merged SCM events : {len(mdata.scm_events_index["merged"])}.')
    for i, (key, value) in enumerate(mdata.scm_events_only_index.items()):
        mdata.logger.info(f'({i + 2}) The number of SCM events only by {key} : {len(value)}.')

    return mdata

def robust_scm_events_integration(mdata: scMetData, cell_marker: dict, main_cell_marker: str, offset: int=1):
    """
    Robustly integrate the results of multiple strategies to obtain merged SCM event.

    :param mdata:
    :param cell_marker:
    :param main_cell_marker:
    :param offset:
    :return:
    """
    index1 = mdata.scm_events_index[main_cell_marker]
    intersection_result = index1.copy()
    scm_events_only_index = {}
    for marker in cell_marker.keys():
        if marker != main_cell_marker:
            index2 = mdata.scm_events_index[marker]
            intersect1_e2, intersect2_e1, diff1_ex_intersect, diff2_ex_intersect = calculate_intersection(index1, index2, offset)
            intersection_result = np.intersect1d(intersection_result, intersect1_e2)
            mdata.scm_events_index[f'merged_{marker}'] = intersect2_e1
            scm_events_only_index[marker] = diff2_ex_intersect

    scm_events_only_index[main_cell_marker] = np.setdiff1d(index1, intersection_result)

    return intersection_result, scm_events_only_index

def calculate_intersection(index1, index2, offset):
    """
    Calculate the intersection of two index sets with a given offset.

    :param index1: The first index set.
    :param index2: The second index set.
    :param offset: The offset value.
    """
    expanded_index1 = np.unique(np.concatenate(
        [index1 - offset, index1, index1 + offset]
    ))
    expanded_index2 = np.unique(np.concatenate(
        [index2 - offset, index2, index2 + offset]
    ))

    intersect1_e2 = np.intersect1d(index1, expanded_index2)
    intersect2_e1 = np.intersect1d(index2, expanded_index1)

    diff1_ex_intersect = np.setdiff1d(index1, intersect1_e2)
    diff2_ex_intersect = np.setdiff1d(index2, intersect2_e1)

    return intersect1_e2, intersect2_e1, diff1_ex_intersect, diff2_ex_intersect


if __name__ == "__main__":
    pass
    # 通过extract_intensity方法，给定代谢物名称和质荷比，根据mz，我们从所有数据点中提取该mz下的丰度值。
    # 应该将使用不同策略从数据中提取单细胞数据的方法 封装成一个函数。 函数的输入是mdata, 输出是mdata中的一个属性scMetEvent