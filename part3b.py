from sklearn.preprocessing import StandardScaler
from scipy.signal import medfilt, savgol_filter
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq
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

def extract_features_group_c(time, force):

    time = np.array(time)
    force = np.array(force)

    X = np.column_stack((time, force))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    time = X_scaled[:, 0]
    force = X_scaled[:, 1]
    
    n = len(time)

    sampling_rate = 100000

    fft_values = fft(force) / n
    frequencies = fftfreq(n, 1 / sampling_rate)

    positive_frequencies = frequencies[:n // 2]
    magnitude_spectrum = np.abs(fft_values[:n // 2])

    peaks, _ = find_peaks(magnitude_spectrum)

    sorted_peaks = peaks[np.argsort(magnitude_spectrum[peaks])][::-1]
    
    first_peak_freq = positive_frequencies[sorted_peaks[0]]
    first_peak_magn = magnitude_spectrum[sorted_peaks[0]]

    second_peak_freq = positive_frequencies[sorted_peaks[1]]
    second_peak_magn = magnitude_spectrum[sorted_peaks[1]]

    return first_peak_freq, first_peak_magn, second_peak_freq, second_peak_magn
