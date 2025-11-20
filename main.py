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


    # plt.figure(figsize=(16,8))
    # plt.title(force_signal)
    # plt.plot(time[80000:], force[80000:])
    # plt.show()

    # Based on exploration we know that the main weld for all experiments starts somewhere after idx 80000

    # We see that leading up to the main weld there is minimal noise. 
    # When we reach the main weld we see a spike in noise

    n = len(time)
    
    window = 50
    noise_threshold_start = 0.025

    ptr = 80000
    while ptr < n:
        
        # Look into a 20 point window
        diff = 1 / window * sum([abs(force[i + 1] - force[i]) for i in range(ptr, ptr + window)])
        if diff >= noise_threshold_start:
            break

        ptr += window

    main_weld_start_idx = ptr

    # When we neear the end of the main weld we see a decrease in noise

    ptr = main_weld_start_idx + 10000  

    noise_threshold_end = 0.1

    while ptr < n:

        diff = 1 / window * sum([abs(force[i + 1] - force[i]) for i in range(ptr, ptr + window)])
        if diff < noise_threshold_end:
            break

        ptr += window

    main_weld_end_idx = ptr

    # plt.figure(figsize=(16,8))
    # plt.title(force_signal)
    # plt.plot(time, force)
    # plt.axvline(x=time[main_weld_start_idx], color='r', linestyle='--')
    # plt.axvline(x=time[main_weld_end_idx], color='r', linestyle='--')
    # plt.legend()
    # plt.show()

power_signals = os.listdir(base_path + "Power_Signals/")

for power_signal in power_signals:

    time = []
    power = []

    with open(base_path + "Power_Signals/" + power_signal, "r") as f:
        
        lines = f.readlines()
        
        for line in lines:
            
            time_, power_ = line.strip().split()
            
            time.append(float(time_))
            power.append(float(power_))

    # plt.figure(figsize=(12,8))
    # plt.title(power_signal)
    # plt.plot(time, power)
    # plt.show()

    # The main weld starts when we observe an increase in power 
    
    n = len(time)

    window = 2
    increase_threshold = 25

    ptr = 0
    while ptr < n:

        increase = 1 / window * sum([power[i + 1] - power[i] for i in range(ptr, ptr + window)])
        if increase >= increase_threshold:
            break

        ptr += window

    main_weld_start_idx = ptr

    # The main weld ends when we observe a decrease in power

    ptr = main_weld_start_idx + 100

    decrease_threshold = 500

    while ptr < n:

        decrease = 1 / window * sum([power[i] - power[i + 1] for i in range(ptr, ptr + window)])
        if decrease >= decrease_threshold:
            break

        ptr += window

    main_weld_end_idx = ptr

    plt.figure(figsize=(12,8))
    plt.title(power_signal)
    plt.plot(time, power)
    plt.axvline(x=time[main_weld_start_idx], color='r', linestyle='--')
    plt.axvline(x=time[main_weld_end_idx], color='r', linestyle='--')
    plt.legend()
    plt.show()