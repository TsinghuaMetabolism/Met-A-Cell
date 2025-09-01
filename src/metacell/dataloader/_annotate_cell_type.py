import os
import bisect
import pandas as pd
import numpy as np
from ._cell_event_extraction import extract_intensity, extract_peak
from .ColorEncoder import  CellTypeMarkerEncoder
from metacell.dataloader.scMetData import scMetData




def annotate_cell_type_by_mz_marker(mdata: scMetData, cell_type_marker, result_path: str = None, offset: int = 1,
                                    sn_ratio: int = 5, interval: int = 6):
    """
    Annotate cell types based on mz markers and extract peak intensities.

    Parameters:
    - mdata (scMetData): The metabolomics data object.
    - cell_type_marker (DataFrame): Dataframe containing cell type markers and their mz values.
    - result_path (str): Path to save figures. Default is None.
    - offset (int): Offset value for annotation. Default is 1.
    - sn_ratio (int): Signal-to-noise ratio threshold for peak extraction. Default is 3.
    - interval (int): Interval for peak extraction. Default is 6.

    Returns:
    - mdata (scMetData): Updated single-cell metabolomics data object with annotated cell types.
    """
    # Create output directory if result_path is provided
    figs_output_dir = os.path.join(result_path, 'cell_type_marker') if result_path else None
    if figs_output_dir:
        os.makedirs(figs_output_dir, exist_ok=True)

    # Update cell type marker dataframe in mdata
    mdata.cell_type_marker_df = cell_type_marker

    # Extract intensity for each marker and append to raw_scm_data
    for name, mz in zip(cell_type_marker['marker_name'], cell_type_marker['mz']):
        # extract the mz and its corresponding intensity by cell type marker.
        mz_intensity = extract_intensity(mdata, name, mz)
        mdata.raw_scm_data.update(mz_intensity)
        new_columns = mz_intensity.loc[:, ~mz_intensity.columns.isin(mdata.raw_scm_data.columns)]
        mdata.raw_scm_data = pd.concat([mdata.raw_scm_data, new_columns], axis=1)
        # extract the signal peaks from the data.
        mdata.cell_type_marker_eic[name], mdata.cell_type_marker_apex_index[name] = extract_peak(mdata, marker=name,
                                                                                                 figs_output_dir=figs_output_dir,
                                                                                                 sn_ratio=sn_ratio,
                                                                                                 interval=interval)
    # Align scm_events with updated raw_scm_data and retain 'CellNumber'
    scm_events = mdata.raw_scm_data.loc[mdata.scm_events.index]
    scm_events.loc[:, 'CellNumber'] = mdata.scm_events['CellNumber']
    mdata.scm_events = scm_events

    # Annotate cell type apex indices with offset
    mdata = annotate_cell_type_apex_index(mdata, offset=offset)

    # Encode and decode cell types
    mdata.cell_type_marker_encoder = CellTypeMarkerEncoder(mdata.cell_type_marker_df)
    mdata.scm_events['cell_type_encode'] = 0
    for marker_name in mdata.cell_type_marker_df['marker_name']:
        increment = mdata.cell_type_marker_encoder.encode(marker_name)
        index_col = f'{marker_name}_index'
        mdata.scm_events['cell_type_encode'] += np.where(~mdata.scm_events[index_col].isna(), increment, 0)

    mdata.scm_events['cell_type'] = mdata.scm_events['cell_type_encode'].apply(mdata.cell_type_marker_encoder.decode)
    mdata.scm_events['cell_type'] = mdata.scm_events['cell_type'].apply(
        lambda x: ','.join(x) if len(x) > 1 else x[0] if x else '')

    mdata.scm_events['cell_type_color'] = mdata.obs['cell_type_encode'].apply(
        mdata.cell_type_marker_encoder.encode_to_color)

    mdata.scm_events['cell_type_name'] = mdata.obs['cell_type_encode'].apply(
        mdata.cell_type_marker_encoder.encode_to_name)

    return mdata


def find_close_elements(list1, list2, offset=1):
    """
    Input: list1 and list2 are two lists of integers.
    The 'offset' parameter specifies the maximum allowable difference (default is 1).

    Condition: For each element x in list1, if there exists an element y in list2 such that |x - y| <= offset,
    then x will be added to the result list.

    Output: A list containing all elements from list1 that meet the condition.
    """
    sorted_list2 = sorted(list2)
    result = []
    for x in list1:
        left = x - offset
        right = x + offset
        # Use binary search to determine if there is an element within the [left, right] range
        idx = bisect.bisect_left(sorted_list2, left)
        if idx < len(sorted_list2) and sorted_list2[idx] <= right:
            result.append(x)
    return(result)


def annotate_cell_type_apex_index(mdata, offset=1):
    """
    Add a column f'{cell_type_marker}_index' to scm_events,
    which records the index values of peaks in cell_type_marker that overlap with SCM peaks.

    Condition: For each element x in scm_events.index, if there exists an element y in list2
    such that |x - y| <= offset, record the value of y.
    """
    scm_events_index = mdata.scm_events.index
    for cell_type_marker in mdata.cell_type_marker_df['marker_name']:
        cell_type_marker_apex_index = sorted(mdata.cell_type_marker_apex_index[cell_type_marker])

        for x in scm_events_index:
            left = x - offset
            right = x + offset
            idx = bisect.bisect_left(cell_type_marker_apex_index, left)
            # 记录满足条件的第一个值，如果存在
            if idx < len(cell_type_marker_apex_index) and cell_type_marker_apex_index[idx] <= right:
                mdata.scm_events.loc[x, f'{cell_type_marker}_index'] = cell_type_marker_apex_index[idx]
    return mdata

