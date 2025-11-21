import numpy as np
import pandas as pd
from scipy.signal import find_peaks as fp

## Functions for Group B

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