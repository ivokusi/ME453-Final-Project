import numpy as np
import pandas as pd
from scipy.signal import find_peaks as fp
import matplotlib.pyplot as plt
from part3a import extract_force_main_weld_segment

## Functions for Group B

## Functions for group C
def extract_peaks(main_signals, **kwargs):
    """
     Input: Main weld signal
     Output : first 2 peaks magnitude and frequency
    """
    for signal in main_signals:
        experiment, time, force = extract_force_main_weld_segment(*signal)
        df_main_weld_signal = pd.DataFrame(data={'Force' : force, 'Time' : time})
        new_df = pd.DataFrame(index = df_main_weld_signal.index)
        for i,vals in df_main_weld_signal.items():
            peaks, _ = fp(vals, **kwargs)
            new_df[i+'_pks'] = new_df.index.isin(peaks+df_main_weld_signal.index.min())
            peak_filter = new_df.sum(axis =1)>=1

        plt.plot(time, force)
        plt.scatter(df_main_weld_signal.loc[peak_filter]['Time'], df_main_weld_signal.loc[peak_filter]['Force'])
        break
        
    return df_main_weld_signal.loc[peak_filter]
        