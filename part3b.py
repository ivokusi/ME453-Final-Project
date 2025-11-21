from scipy.signal import medfilt, savgol_filter
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

## Functions for Group B

def denoise_power(power):

  power_med = medfilt(power, kernel_size=7)
  return savgol_filter(power_med, window_length=51, polyorder=3, mode='interp')

def extract_features_group_b(time, power):

    """
    Returns the risePeak, riseSlope, riseDuration, and dipDepth of the main weld section
    """

    # Remove noise
    denoised_power = denoise_power(power) 

    n = len(time)

    window = 2
    peak_threshold = 5

    ptr = 20 # many experiments start with a small downslope before a rise, this helps
    while ptr < n:

        # peak threshold condition

        diff = 1 / window * sum([abs(denoised_power[i + 1] - denoised_power[i]) for i in range(ptr, ptr + window)])
        
        if diff <= peak_threshold:
            break

        # actual peak condition

        curr_window_avg = 1 / window * sum([denoised_power[i] for i in range(ptr, ptr + window)])
        next_window_avg = 1 / window * sum([denoised_power[i] for i in range(ptr + 1, ptr + window + 1)])

        if next_window_avg < curr_window_avg:
            break

        ptr += 1

    rise_peak = power[ptr]
    rise_slope = (power[ptr] - power[0]) / time[ptr] - time[0]
    rise_duration = time[ptr] - time[0]

    dip = min(power[ptr:n])

    dip_depth = rise_peak - dip
    
    # plt.figure(figsize=(16,8))
    # plt.plot(time, denoised_power)
    # plt.hlines(dip, xmin=time[0], xmax=time[-1], color="red")
    # plt.show()

    return rise_peak, rise_slope, rise_duration, dip_depth

## Functions for group C
def extract_peaks(main_signal, **kwargs):
    """
     Input: Main weld signal
     Output : first 2 peaks magnitude and frequency
    """
    time, force = main_signal
    df_main_weld_signal = pd.DataFrame(data={'Force' : force, 'Time' : time})
    print(df_main_weld_signal)
    """
    new_df = pd.DataFrame(index = df_main_weld_signal.index)
    for i,vals in df_main_weld_signal.items():
        peaks, _ = fp(vals, **kwargs)
        new_df[i+'_pks'] = new_df.index.isin(peaks+df_main_weld_signal.index.min())
        peak_filter = new_df.sum(axis =1)>=1
    return df_main_weld_signal.loc[peak_filter]
    """