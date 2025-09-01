import numpy as np
import pandas as pd
from tqdm import tqdm
from ._utils import sliding_window_baseline_correction, detect_peaks, calculate_signal_threshold_from_signal
from ._utils import double_scan_feature_integration, calculate_intersection

def extract_features_by_peak_matching(mdata, mz_list):
    """
    """
    mdata.logger.info("Start extracting features by peak matching.")
    mdata.cell_feature_matrix= extract_feature_peaks(mdata, mz_list, ppm_threshold=10, sn_ratio=3, interval=6, offset=1)
    mdata.logger.info("Finish extracting features by peak matching.")
    mdata.processing_status['feature_extraction_strategy'] = 'peak matching'
    return mdata


def extract_feature_peaks(mdata, mz_list, ppm_threshold: float =10, sn_ratio: int =3, interval: int =6, offset: int=1):
    mz_data = mdata.mz_data
    intensity_data = mdata.intensity_data
    scan_start_time = mdata.raw_scm_data['scan_start_time']
    scm_events_index= np.array(mdata.scm_events.index)

    cell_feature_matrix = pd.DataFrame(
        index = np.arange(1 ,len(mz_list) +1),
        columns = mdata.scm_events['CellNumber']
    )

    flag = 0
    cluster_info = pd.DataFrame()
    for mz in tqdm(mz_list, total=len(mz_list), desc='Extracting features by peak matching'):
        flag = flag + 1
        density_center, mz_intensity = double_scan_feature_integration(mz_data, intensity_data, mz, ppm_threshold)

        df = sliding_window_baseline_correction(mz_intensity['intensity'], scan_start_time, sn_ratio, output2figures=None, window_size=100, p=0.5)

        mph = calculate_signal_threshold_from_signal(df, multiplier = sn_ratio)
        peaks = detect_peaks(df['signal'], mpd=interval, mph=mph)

        intersect1_e2, intersect2_e1, _, _ = calculate_intersection(scm_events_index, peaks, offset)

        cellnumber = len(intersect1_e2)
        cellratio = cellnumber / len(scm_events_index) * 100

        mz_mean = mz_intensity.loc[intersect2_e1, 'mz'].mean()
        mz_median = mz_intensity.loc[intersect2_e1, 'mz'].median()


        info = {'Feature': flag, 'mz_center': mz, 'mz_mean': mz_mean, 'mz_median': mz_median,
                'hits': cellnumber, 'hit_rate': cellratio}
        cluster_info = pd.concat([cluster_info, pd.DataFrame([info])], ignore_index=True)

        for idx1, idx2 in zip(intersect1_e2, intersect2_e1):
            cell_number = mdata.scm_events.loc[idx1, 'CellNumber']
            intensity = mz_intensity.loc[idx2, 'intensity']
            cell_feature_matrix.loc[flag, cell_number] = intensity

    cluster_info.set_index('Feature', inplace=True)
    cell_feature_matrix = cluster_info.join(cell_feature_matrix)
    cell_feature_matrix = cell_feature_matrix.reset_index().rename(columns={'index': 'Feature'})
    cell_feature_matrix.sort_values(by='hit_rate', ascending=False, inplace=True)

    return cell_feature_matrix