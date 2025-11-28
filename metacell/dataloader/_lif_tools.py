import numpy as np
from ._utils import sliding_window_baseline_correction, detect_peaks, calculate_signal_threshold_from_signal
from ._LIF_MS_integrator import check_lif_peak

def extract_lif_peak(lif_data, interval=60, sn_ratio=3, window_size=1000):
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
    df = sliding_window_baseline_correction(lif_data['intensity'], lif_data['times'], sn_ratio,
                                                    output2figures=None, window_size=window_size, p=0.5)
    # Detect peaks in the intensity data
    mph = calculate_signal_threshold_from_signal(df, multiplier=sn_ratio)
    peaks = detect_peaks(lif_data['intensity'], mpd=interval, mph=mph)

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
    lif_peak_data = lif_peak_data[lif_peak_data['corrected_intensity'] > 0]
    
    return lif_peak_data

