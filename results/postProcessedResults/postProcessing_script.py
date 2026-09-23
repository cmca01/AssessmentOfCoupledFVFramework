#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 14:17:03 2025

@author: colmmcalister
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
import os

# Define input and output directories relative to this script. The script and
# generated plots live in ``results/postProcessedResults``; input data lives in
# the sibling ``results/rawResults`` directory.
script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.dirname(script_dir)
raw_results_dir = os.path.join(results_dir, "rawResults")
sim_dir = os.path.join(raw_results_dir, "numericalModelResults/F3A3")
exp_dir = os.path.join(raw_results_dir, "experimentResults")


#--- TIME WINDOW                                                                                                                                                            
t_window_start = 18
t_window_end = 24

#--- SAVE PLOTS AS PDF
save_wave_elevation_pdf = False
save_pitch_pdf = False
save_surge_pdf = False
save_heave_pdf = False
save_front_tension_pdf = False
save_rear_tension_pdf = False

save_wave_elevation_window_pdf = False
save_pitch_window_pdf = False
save_surge_window_pdf = False
save_heave_window_pdf = False
save_front_tension_window_pdf = False
save_rear_tension_window_pdf = False

save_pitch_fft_window_pdf = False
save_surge_fft_window_pdf = False
save_heave_fft_window_pdf = False
save_wave_probe_fft_window_pdf = False
save_front_tension_fft_window_pdf = False
save_rear_tension_fft_window_pdf = False


# Automatically build full paths to the four required files                                                                                                                 
sim_file = os.path.join(sim_dir, "sixDoF_History/0/sixDoFRigidBodyStateFvBeam.dat")
wave_sim_file = os.path.join(sim_dir, "interfaceHeight1/0/height.dat")
front_tension_file = os.path.join(sim_dir, "0/anchorForcebeamone.dat")
rear_tension_file = os.path.join(sim_dir, "0/anchorForcebeamtwo.dat")

# The old results layout ended in ``.../<case>/postProcessing``, whereas the
# assessment results use ``.../numericalModelResults/<case>`` directly.
sim_path = os.path.normpath(sim_dir)
folder_name = os.path.basename(sim_path)
if folder_name.lower() == "postprocessing":
    folder_name = os.path.basename(os.path.dirname(sim_path))

match_simple = re.search(r"F(\d+(?:\.\d+)?)A(\d+(?:\.\d+)?)", folder_name)

if match_simple:
    # Convert F5A6 → frequency 0.5 Hz, amplitude 6 cm                                                                                                                       
    raw_freq = float(match_simple.group(1))
    frequency = raw_freq / 10 if raw_freq >= 1 else raw_freq  # turn 5 → 0.5, 10→1.0 etc.                                                                                   
    amplitude = float(match_simple.group(2)) / 100  # cm → m                                                                                                                
else:
    raise ValueError(f"❌ Could not parse frequency/amplitude from folder name: {folder_name}")

# Keep plots for each simulation case in their own directory, e.g.
# results/postProcessedResults/F3A3_plots.
case_name = match_simple.group(0)
plot_dir = os.path.join(script_dir, f"{case_name}_plots")
os.makedirs(plot_dir, exist_ok=True)


exp_filename = f"{frequency}hz_{int(amplitude*100)}cm.csv"
exp_file = os.path.join(exp_dir, exp_filename)

if not os.path.exists(exp_file):
    raise FileNotFoundError(f"❌ Experiment file not found: {exp_file}")




# 1) Extract frequency & amplitude from experimental filename
exp_filename = exp_file.split("/")[-1]
m = re.search(r"([\d.]+)hz_([\d.]+)cm", exp_filename, flags=re.IGNORECASE)
if m:
    frequency = float(m.group(1))
    amplitude = float(m.group(2)) / 100.0
else:
    frequency = amplitude = None

# 2) Extract E base and exponent from simulation folder name
sim_folder = folder_name  # e.g. "F0.3Hz_A3cm_E4.734_10^5Pa"
m2 = re.search(r"E([\d.]+)(?:_10\^([-\d]+))?Pa", sim_folder)
if m2:
    E_base = float(m2.group(1))         # e.g. 4.734
    exp_str = m2.group(2) or "0"        # e.g. "5" or default "0"
    exponent = int(exp_str)             # e.g. 5
    E_value = E_base * 10**exponent     # numeric value in Pa
else:
    E_base = exponent = E_value = None

# 3) Read & parse simulation data
# Regex to extract simulation data
forceRegex = r"([0-9.Ee\-+]+)\s+\(+([0-9.Ee\-+]+)\s([0-9.Ee\-+]+)\s([0-9.Ee\-+]+)\)\s\(([0-9.Ee\-+]+)\s([0-9.Ee\-+]+)\s([0-9 .Ee\-+]+)\)+\s\(+([0-9.Ee\-+]+)\s([0-9.Ee\-+]+)\s([0-9.Ee\-+]+)\)\s\(([0-9.Ee\-+]+)\s([0-9.Ee\-+]+)\s([0-9.Ee\-+]+)\)+"

# Initialize lists for simulation data
t_sim, surge_sim, sway_sim, heave_sim = [], [], [], []
force_x_sim, force_y_sim, force_z_sim = [], [], []
roll_sim, pitch_sim, yaw_sim = [], [], []

# Read and parse simulation file
with open(sim_file, 'r') as pipefile:
    for line in pipefile:
        match = re.search(forceRegex, line)
        if match:
            t_sim.append(float(match.group(1)))
            surge_sim.append(float(match.group(2)))   # Surge (X)
            sway_sim.append(float(match.group(3)))    # Sway (Y)
            heave_sim.append(float(match.group(4)))   # Heave (Z)
            force_x_sim.append(float(match.group(5))) # Force in X
            force_y_sim.append(float(match.group(6))) # Force in Y
            force_z_sim.append(float(match.group(7))) # Force in Z
            roll_sim.append(float(match.group(8)))    # Roll (X-axis rotation)
            pitch_sim.append(float(match.group(9)))   # Pitch (Y-axis rotation)
            yaw_sim.append(float(match.group(10)))    # Yaw (Z-axis rotation)

# Convert lists to NumPy arrays
t_sim, surge_sim, heave_sim, pitch_sim = map(np.array, [t_sim, surge_sim, heave_sim, pitch_sim])

# Sort simulation data by time
sorted_indices = np.argsort(t_sim)
t_sim, surge_sim, heave_sim, pitch_sim = (
    t_sim[sorted_indices], surge_sim[sorted_indices], heave_sim[sorted_indices], pitch_sim[sorted_indices]
)

# Modify simulation data
heave_sim -= 0.6      # Adjust heave relative to water level at 0.7m
surge_sim -= 5.75     # Adjust surge starting position
t_sim += 1


# 0) Read your new wave-simulation data
#    adjust delimiter / skiprows as needed
sim_wave = np.loadtxt(wave_sim_file)
t_wave_sim = sim_wave[:, 0]      # first column: time
wave_height_sim = sim_wave[:, 1] # second column: height

wave_height_sim -= 0.701 #adjust
t_wave_sim += 2

h_offset = -2

pitch_rad = pitch_sim * np.pi/180 # convert pitch into radians

heave_sim_corrected = heave_sim + h_offset #* (np.cos(pitch_rad) - 1)
surge_sim_corrected = surge_sim - h_offset*(np.sin(pitch_rad))


# Read experimental data (no headers)
df_exp = pd.read_csv(exp_file, header=None)

# Extract relevant columns based on their indices (0-based indexing)
t_exp = df_exp.iloc[:, 0].values   # Column 1 (Time)
surge_exp = df_exp.iloc[:, 4].values * 0.001  # Column 5 (Surge)
heave_exp = df_exp.iloc[:, 6].values * 0.001  # Column 7 (Heave)
pitch_exp = df_exp.iloc[:, 8].values  # Column 9 (Pitch)
waveprobe_exp = df_exp.iloc[:,3].values # Column 4 (Wave Elevation relative to Mean Water Level Surface)

pitch_sim = np.degrees(pitch_sim)

## Remove NaNs from simulation data
valid_pitch_sim = ~np.isnan(t_sim) & ~np.isnan(pitch_sim)
pitch_sim_clean = pitch_sim[valid_pitch_sim]
t_pitch_sim_clean = t_sim[valid_pitch_sim]

valid_surge_sim = ~np.isnan(t_sim) & ~np.isnan(surge_sim)
surge_sim_clean = surge_sim[valid_surge_sim]
t_surge_sim_clean = t_sim[valid_surge_sim]

valid_heave_sim = ~np.isnan(t_sim) & ~np.isnan(heave_sim_corrected)
heave_sim_clean = heave_sim[valid_heave_sim]
t_heave_sim_clean = t_sim[valid_heave_sim]

valid_wave_sim = ~np.isnan(t_wave_sim) & ~np.isnan(wave_height_sim)
wave_sim_clean = wave_height_sim[valid_wave_sim]
t_wave_sim_clean = t_wave_sim[valid_wave_sim]

# Remove NaNs from experimental data
valid_pitch_exp = ~np.isnan(pitch_exp)
pitch_exp_clean = pitch_exp[valid_pitch_exp]
t_pitch_exp_clean = t_exp[valid_pitch_exp]

valid_surge_exp = ~np.isnan(surge_exp)
surge_exp_clean = surge_exp[valid_surge_exp]
t_surge_exp_clean = t_exp[valid_surge_exp]

valid_heave_exp = ~np.isnan(heave_exp)
heave_exp_clean = heave_exp[valid_heave_exp]
t_heave_exp_clean = t_exp[valid_heave_exp]

valid_wave_exp = ~np.isnan(waveprobe_exp)
wave_exp_clean = waveprobe_exp[valid_wave_exp]
t_wave_exp_clean = t_exp[valid_wave_exp]


# Load data for beamone (skip the first row)
df_anchor1 = np.loadtxt(front_tension_file, skiprows=1)
df_anchor1_sorted = df_anchor1[np.argsort(df_anchor1[:, 0])]

# Extract time values and force magnitudes for beamone
x_values1 = df_anchor1_sorted[:, 0]  
y_values1 = np.sqrt(df_anchor1_sorted[:, 1]**2 + df_anchor1_sorted[:, 2]**2 + df_anchor1_sorted[:, 3]**2)

# Load data for beamtwo (skip the first row)
df_anchor2 = np.loadtxt(rear_tension_file, skiprows=1)
df_anchor2_sorted = df_anchor2[np.argsort(df_anchor2[:, 0])]

# Extract time values and force magnitudes for beamtwo
x_values2 = df_anchor2_sorted[:, 0]  
y_values2 = np.sqrt(df_anchor2_sorted[:, 1]**2 + df_anchor2_sorted[:, 2]**2 + df_anchor2_sorted[:, 3]**2)

# Read experimental data (no headers)
df_exp = pd.read_csv(exp_file, header=None)


# Extract frequency and amplitude from filename
exp_filename = exp_file.split("/")[-1]  # Get only the filename
match = re.search(r"([\d.]+)hz_([\d.]+)cm", exp_filename)
if match:
    frequency = float(match.group(1))  # Convert to float
    amplitude = float(match.group(2)) / 100  # Convert cm to meters
else:
    frequency, amplitude = None, None  # Default if not found
  
# Extract relevant columns based on their indices (0-based indexing)
t_exp = df_exp.iloc[:, 0].values   # Column 1 (Time)
front_exp = df_exp.iloc[:, 1].values  # Column 11 Front Load Cell
rear_exp = df_exp.iloc[:, 2].values # Column 12 Rear Load Cell

t_exp = df_exp.iloc[:, 0].values   # Column 1 (Time)


length_time_array= 200

# Choose common time range and resolution
time_common = np.linspace(
    max(t_wave_exp_clean[0], t_wave_sim_clean[0]),
    min(t_wave_exp_clean[-1], t_wave_sim_clean[-1]),
    num=length_time_array
)


#############

# 2) Extract E base and exponent from simulation folder name
sim_folder = folder_name  # e.g. "F0.3Hz_A3cm_E4.734_10^5Pa"
m2 = re.search(r"E([\d.]+)(?:_10\^([-\d]+))?Pa", sim_folder)
if m2:
    E_base = float(m2.group(1))         # e.g. 4.734
    exp_str = m2.group(2) or "0"        # e.g. "5" or default "0"
    exponent = int(exp_str)             # e.g. 5
    E_value = E_base * 10**exponent     # numeric value in Pa
else:
    E_base = exponent = E_value = None

def build_title(varname):
    base = f"{varname} (f = {frequency:.3f} Hz, A = {amplitude:.3f} m"
    if E_base is not None:
        base += f", E = {E_base}×10^{exponent} Pa"
    return base + ")"

### SMOOTHING ####

# Moving Average Function
def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Apply Savitzky-Golay filter
window_size = 11  # Must be odd
poly_order = 1    # Polynomial order
front_exp_smooth = savgol_filter(front_exp, window_size, poly_order)
rear_exp_smooth = savgol_filter(rear_exp, window_size, poly_order)

####### 

### Shift Time ###
t_shift = 1.6
x_values1 = x_values1 + t_shift
x_values2 = x_values2 + t_shift




def plot_signal_comparison(t_exp, signal_exp, t_sim, signal_sim,
                           quantity_name, unit='',
                           t_start=None, t_end=None,
                           save_pdf=False, pdf_filename=None):
    """
    Creates a time-series comparison plot between experiment and simulation.

    Parameters
    ----------
    t_exp, signal_exp : arrays
        Experimental time and signal.
    t_sim, signal_sim : arrays
        Simulation time and signal.
    quantity_name : str
        Name of the quantity (for plot title / labels).
    unit : str
        Unit for y-axis.
    t_start, t_end : float
        Optional time window for slicing.
    """

    # Optional time window slicing
    if t_start is not None and t_end is not None:
        exp_mask = (t_exp >= t_start) & (t_exp <= t_end)
        sim_mask = (t_sim >= t_start) & (t_sim <= t_end)
        t_exp = t_exp[exp_mask]
        signal_exp = signal_exp[exp_mask]
        t_sim = t_sim[sim_mask]
        signal_sim = signal_sim[sim_mask]

    plt.figure(figsize=(10, 5))
    plt.plot(t_exp, signal_exp, label="Experiment", color='green', linestyle='-.', linewidth=2)
    plt.plot(t_sim, signal_sim, label="Simulation", color='blue', linewidth=2)

    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel(f"{quantity_name} ({unit})", fontsize=12)

    title = f"{quantity_name} – Time Series"
    if t_start is not None:
        title += f" ({t_start}–{t_end} s)"
    #plt.title(title)

    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    if save_pdf and pdf_filename is not None:
        plt.savefig(os.path.join(plot_dir, pdf_filename))
    plt.show()



##################################
###       FFTs                 ###
##################################
def plot_fft_comparison(t_exp, signal_exp, t_sim, signal_sim, quantity_name, unit='', save_pdf=False, pdf_filename=None):
    """
    Plots the FFT comparison of experimental and simulation signals for a given quantity.
    
    Parameters:
    - t_exp: time vector for experimental data
    - signal_exp: experimental signal (e.g., wave probe, tension)
    - t_sim: time vector for simulation data
    - signal_sim: simulated signal
    - quantity_name: name of the quantity (for title/label)
    - unit: unit of the quantity (for y-axis)
    """
    def compute_fft(t, signal):
        N = len(t)
        dt = t[1] - t[0]
        yf = np.fft.fft(signal)
        xf = np.fft.fftfreq(N, d=dt)
        pos_mask = xf > 0
        xf_pos = xf[pos_mask]
        yf_pos = yf[pos_mask]
        amplitudes = (2.0 / N) * np.abs(yf_pos)
        return xf_pos, amplitudes

    # Compute FFTs
    xf_exp, amp_exp = compute_fft(t_exp, signal_exp)
    xf_sim, amp_sim = compute_fft(t_sim, signal_sim)
    
    amp_max_exp= max(amp_exp)
    amp_max_sim= max(amp_sim)
    
    # Experiment
    idx_max_exp = np.argmax(amp_exp)
    amp_max_exp = amp_exp[idx_max_exp]
    freq_max_exp = xf_exp[idx_max_exp]
     
    # Simulation
    idx_max_sim = np.argmax(amp_sim)
    amp_max_sim = amp_sim[idx_max_sim]
    freq_max_sim = xf_sim[idx_max_sim]
    
    percentage_error = ((amp_max_sim - amp_max_exp)/amp_max_exp) * 100

    print(quantity_name+" FFT Peaks:")
    print(f"Experimental: amplitude = {amp_max_exp:.3f} "+unit+f", frequency = {freq_max_exp:.3f} Hz")
    print(f"Simulation: amplitude = {amp_max_sim:.3f} "+unit+f", frequency = {freq_max_sim:.3f} Hz")
    print(f"Percentage Error= {percentage_error:.3f} %")
    print(" ")
    


    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(xf_exp, amp_exp, label="Experiment", color='green', linestyle='-.', linewidth=2)
    plt.plot(xf_sim, amp_sim, label="Simulation", color='blue', linewidth=2)
    plt.xlabel("Frequency (Hz)", fontsize=12)
    plt.ylabel(f"Amplitude ({unit})", fontsize=12)
    plt.title(f"FFT of {quantity_name}")
    plt.axhline(amp_max_exp, color= 'g', linestyle='-.', label= f"Experiment Max Amplitude = {amp_max_exp:.3f} {unit}")
    plt.axhline(amp_max_sim,  color= 'b', linestyle='-', label= f"Simulation Max Amplitude = {amp_max_sim:.3f} {unit}")
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.xlim(0,5)
    if save_pdf and pdf_filename is not None:
        plt.savefig(os.path.join(plot_dir, pdf_filename))
    plt.show()
 


def plot_fft_comparison_new(t_exp, signal_exp, t_sim, signal_sim, N_samples, quantity_name, unit='', save_pdf=False, pdf_filename=None):
    """
    Plots the FFT comparison of experimental and simulation signals for a given quantity.
    """
    def compute_fft(t, signal):
        N = len(t)
        dt = t[1] - t[0]
        yf = np.fft.fft(signal)
        xf = np.fft.fftfreq(N, d=dt)
        pos_mask = xf > 0
        xf_pos = xf[pos_mask]
        yf_pos = yf[pos_mask]
        amplitudes = (2.0 / N) * np.abs(yf_pos)
        return xf_pos, amplitudes



    def signal_process(t, signal, N):
        """
        Resample the input signal to have exactly N points.
        
        Parameters
        ----------
        t : array-like
            Original time array.
        signal : array-like
            Original signal values.
        N : int
            Desired number of output points.
        
        Returns
        -------
        t_new : ndarray
            Resampled time array with N points.
        signal_new : ndarray
            Resampled signal array with N points.
        """
        # Create interpolation function
        interp_func = interp1d(t, signal, kind='linear', fill_value="extrapolate")
        
        # Generate new evenly spaced time array
        t_new = np.linspace(t[0], t[-1], N)
        
        # Interpolate signal to match new time array
        signal_new = interp_func(t_new)
        
        return t_new, signal_new
   
    
    
    t_sim, signal_sim = signal_process(t_sim, signal_sim, N_samples)
    t_exp, signal_exp = signal_process(t_exp, signal_exp, N_samples)
    

    # Compute FFTs
    xf_exp, amp_exp = compute_fft(t_exp, signal_exp)
    xf_sim, amp_sim = compute_fft(t_sim, signal_sim)
    
    
   # Experiment
    idx_max_exp = np.argmax(amp_exp)
    amp_max_exp = amp_exp[idx_max_exp]
    freq_max_exp = xf_exp[idx_max_exp]
    
    # Simulation
    idx_max_sim = np.argmax(amp_sim)
    amp_max_sim = amp_sim[idx_max_sim]
    freq_max_sim = xf_sim[idx_max_sim]
    
    percentage_error = ((amp_max_sim - amp_max_exp)/amp_max_exp) * 100
    
    print(quantity_name+" FFT Peaks:")
    print(f"Experimental: amplitude = {amp_max_exp:.3f} "+unit+f", frequency = {freq_max_exp:.3f} Hz")
    print(f"Simulation: amplitude = {amp_max_sim:.3f} "+unit+f", frequency = {freq_max_sim:.3f} Hz")
    print(f"Percentage Error= {percentage_error:.3f} %")
    print(" ")
    
    
    
    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(xf_exp, amp_exp, label="Experiment", color='green', linestyle='-.', linewidth=2)
    plt.plot(xf_sim, amp_sim, label="Simulation", color='blue', linewidth=2)
    plt.xlabel("Frequency (Hz)", fontsize=12)
    plt.ylabel(f"Amplitude ({unit})", fontsize=12)
    plt.title(f"FFT of {quantity_name}")
    plt.axhline(amp_max_exp, color= 'g', linestyle='-.', label= f"Experiment Max Amplitude = {amp_max_exp:.3f} {unit}")
    plt.axhline(amp_max_sim,  color= 'b', linestyle='-', label= f"Simulation Max Amplitude = {amp_max_sim:.3f} {unit}")
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.xlim(0,5)
    if save_pdf and pdf_filename is not None:
        plt.savefig(os.path.join(plot_dir, pdf_filename))
    plt.show()

    
#--- TIME WINDOW

def slice_time_window(t, y, t_start, t_end):
    mask = (t >= t_start) & (t <= t_end)
    return t[mask], y[mask]

# Slice all datasets
t_wave_exp_win, wave_exp_win = slice_time_window(t_wave_exp_clean, wave_exp_clean, t_window_start, t_window_end)
t_wave_sim_win, wave_sim_win = slice_time_window(t_wave_sim, wave_height_sim, t_window_start, t_window_end)

t_pitch_exp_win, pitch_exp_win = slice_time_window(t_pitch_exp_clean, pitch_exp_clean, t_window_start, t_window_end)
t_pitch_sim_win, pitch_sim_win = slice_time_window(t_sim, pitch_sim, t_window_start, t_window_end)

t_surge_exp_win, surge_exp_win = slice_time_window(t_surge_exp_clean, surge_exp_clean, t_window_start, t_window_end)
t_surge_sim_win, surge_sim_win = slice_time_window(t_sim, surge_sim, t_window_start, t_window_end)

t_heave_exp_win, heave_exp_win = slice_time_window(t_heave_exp_clean, heave_exp_clean, t_window_start, t_window_end)
t_heave_sim_win, heave_sim_win = slice_time_window(t_sim, heave_sim, t_window_start, t_window_end)

t_front_exp_win, front_exp_win = slice_time_window(t_exp, front_exp_smooth, t_window_start, t_window_end)
t_front_sim_win, front_sim_win = slice_time_window(x_values1, y_values1, t_window_start, t_window_end)

t_rear_exp_win, rear_exp_win = slice_time_window(t_exp, rear_exp_smooth, t_window_start, t_window_end)
t_rear_sim_win, rear_sim_win = slice_time_window(x_values2, y_values2, t_window_start, t_window_end)


N_samples = 500



#--------------------
# FULL TIME PLOTS
#-------------------

# Wave Elevation
plot_signal_comparison(
    t_wave_exp_clean, wave_exp_clean,
    t_wave_sim_clean, wave_height_sim,
    "Wave Elevation", "m",
    save_pdf=save_wave_elevation_pdf,
    pdf_filename="wave_elevation.pdf"
)

# Pitch
plot_signal_comparison(
    t_pitch_exp_clean, pitch_exp_clean,
    t_pitch_sim_clean, pitch_sim,
    "Pitch", "°",
    save_pdf=save_pitch_pdf,
    pdf_filename="pitch.pdf"
)

# Surge
plot_signal_comparison(
    t_surge_exp_clean, surge_exp_clean,
    t_surge_sim_clean, surge_sim_corrected,
    "Surge", "m",
    save_pdf=save_surge_pdf,
    pdf_filename="surge.pdf"
)

# Heave
plot_signal_comparison(
    t_heave_exp_clean, heave_exp_clean,
    t_heave_sim_clean, heave_sim_corrected,
    "Heave", "m",
    save_pdf=save_heave_pdf,
    pdf_filename="heave.pdf"
)

# Front Tension
plot_signal_comparison(
    t_exp, front_exp_smooth,
    x_values1, y_values1,
    "Front Tension", "N",
    save_pdf=save_front_tension_pdf,
    pdf_filename="front_tension.pdf"
)

# Rear Tension
plot_signal_comparison(
    t_exp, rear_exp_smooth,
    x_values2, y_values2,
    "Rear Tension", "N",
    save_pdf=save_rear_tension_pdf,
    pdf_filename="rear_tension.pdf"
)


#--------------------
# Time Window Plots
#-------------------
# Wave Elevation
plot_signal_comparison(
    t_wave_exp_win, wave_exp_win,
    t_wave_sim_win, wave_sim_win,
    "Wave Elevation", "m",
    t_start=t_window_start, t_end=t_window_end,
    save_pdf=save_wave_elevation_window_pdf,
    pdf_filename="wave_elevation_window.pdf"
)

# Pitch
plot_signal_comparison(
    t_pitch_exp_win, pitch_exp_win,
    t_pitch_sim_win, pitch_sim_win,
    "Pitch", "°",
    t_start=t_window_start, t_end=t_window_end,
    save_pdf=save_pitch_window_pdf,
    pdf_filename="pitch_window.pdf"
)

# Surge
plot_signal_comparison(
    t_surge_exp_win, surge_exp_win,
    t_surge_sim_win, surge_sim_win,
    "Surge", "m",
    t_start=t_window_start, t_end=t_window_end,
    save_pdf=save_surge_window_pdf,
    pdf_filename="surge_window.pdf"
)

# Heave
plot_signal_comparison(
    t_heave_exp_win, heave_exp_win,
    t_heave_sim_win, heave_sim_win,
    "Heave", "m",
    t_start=t_window_start, t_end=t_window_end,
    save_pdf=save_heave_window_pdf,
    pdf_filename="heave_window.pdf"
)

# Front Tension
plot_signal_comparison(
    t_front_exp_win, front_exp_win,
    t_front_sim_win, front_sim_win,
    "Front Tension", "N",
    t_start=t_window_start, t_end=t_window_end,
    save_pdf=save_front_tension_window_pdf,
    pdf_filename="front_tension_window.pdf"
)

# Rear Tension
plot_signal_comparison(
    t_rear_exp_win, rear_exp_win,
    t_rear_sim_win, rear_sim_win,
    "Rear Tension", "N",
    t_start=t_window_start, t_end=t_window_end,
    save_pdf=save_rear_tension_window_pdf,
    pdf_filename="rear_tension_window.pdf"
)


"""
#--------------------
# FULL TIME FFTs
#-------------------

# Waveprobe
plot_fft_comparison(t_exp, waveprobe_exp, t_wave_sim, wave_height_sim, "Wave Probe", "m")

# Pitch
plot_fft_comparison(t_pitch_exp_clean, pitch_exp_clean, t_sim, pitch_sim, "Pitch", "°")

# Surge
plot_fft_comparison(t_surge_exp_clean, surge_exp_clean, t_sim, surge_sim_corrected, "Surge", "m")

# Heave
plot_fft_comparison(t_heave_exp_clean, heave_exp_clean, t_sim, heave_sim_corrected, "Heave", "m")

# Front Tension
plot_fft_comparison(t_exp, front_exp, x_values1, y_values1, "Front Tension", "N")

# Rear Tension
plot_fft_comparison(t_exp, rear_exp, x_values2, y_values2, "Rear Tension", "N")
"""

#--------------------
# TIME WINDOW FFTs
#-------------------

# Pitch
plot_fft_comparison(t_pitch_exp_win, pitch_exp_win, t_pitch_sim_win, pitch_sim_win, f"Pitch (Time Window: {t_window_start}s to {t_window_end}s)", "°", save_pdf=save_pitch_fft_window_pdf, pdf_filename="pitch_fft_window.pdf")

# Surge
plot_fft_comparison(t_surge_exp_win, surge_exp_win, t_surge_sim_win, surge_sim_win, f"Surge (Time Window: {t_window_start}s to {t_window_end}s)", "m", save_pdf=save_surge_fft_window_pdf, pdf_filename="surge_fft_window.pdf")

# Heave
plot_fft_comparison(t_heave_exp_win, heave_exp_win, t_heave_sim_win, heave_sim_win, f"Heave (Time Window: {t_window_start}s to {t_window_end}s)", "m", save_pdf=save_heave_fft_window_pdf, pdf_filename="heave_fft_window.pdf")

# Waveprobe
plot_fft_comparison(t_wave_exp_win, wave_exp_win, t_wave_sim_win, wave_sim_win, f"Wave Probe (Time Window: {t_window_start}s to {t_window_end}s)", "m", save_pdf=save_wave_probe_fft_window_pdf, pdf_filename="wave_probe_fft_window.pdf")

# Front Tension
plot_fft_comparison_new(t_front_exp_win, front_exp_win, t_front_sim_win, front_sim_win, N_samples, f"Front Tension (Time Window: {t_window_start}s to {t_window_end}s)", "N", save_pdf=save_front_tension_fft_window_pdf, pdf_filename="front_tension_fft_window.pdf")

# Rear Tension
plot_fft_comparison_new(t_rear_exp_win, rear_exp_win, t_rear_sim_win, rear_sim_win, N_samples, f"Rear Tension (Time Window: {t_window_start}s to {t_window_end}s)", "N", save_pdf=save_rear_tension_fft_window_pdf, pdf_filename="rear_tension_fft_window.pdf")
