import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os

# loop through files within Force_Signals and Power_Signals dir

base_path = "./data/part3/"

force_signals = os.listdir(base_path + "Force_Signals/")

for force_signal in force_signals:
    
    time = []
    force = []
    
    with open(base_path + "Force_Signals/" + force_signal, "r") as f:
        
        lines = f.readlines()
        
        for line in lines:
            
            time_, force_ = line.strip().split()
            
            time.append(float(time_))
            force.append(float(force_))


    plt.figure(figsize=(16,8))
    plt.title(force_signal)
    plt.plot(time[80000:], force[80000:])
    plt.show()

# For each file data is in csv-style document where delimiter is " " (whitespace). The first column corresponds to time and the second to the metric (Force or Power)
