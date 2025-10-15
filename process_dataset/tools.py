import numpy as np
from scipy.signal import butter, filtfilt, buttord
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.interpolate import interp1d
import math 
import sys
import neurokit2 as nk
import random


def butterworth_lowpass_filter(signal, fs, passband=50, stopband=60, 
                              pass_ripple=1.0, stop_atten=2.5, order=None):
    nyq = 0.5 * fs
    wp = passband / nyq
    ws = stopband / nyq
    
    if order is None:
        N, Wn = buttord(wp, ws, pass_ripple, stop_atten)
    else:
        N = order
        Wn = passband / nyq
    # print(f"Order: {N}")  
    b, a = butter(N, Wn, btype='low')
    filtered_signal = filtfilt(b, a, signal)
    
    return filtered_signal
  
def resample_signal_with_rpeaks(signal, original_fs, new_fs):
    signal = np.array(signal)
    N = len(signal)
    _, rpeaks_info = nk.ecg_peaks(signal, sampling_rate=original_fs)
    r_peaks_original = rpeaks_info["ECG_R_Peaks"]
    if original_fs == new_fs:
        return signal, r_peaks_original
    else:
        t_original = np.arange(N) / original_fs
        
        interpolator = interp1d(t_original, signal, kind='cubic', fill_value="extrapolate")
        duration = N / original_fs
        t_new = np.arange(0, duration, 1/new_fs)
        new_signal = interpolator(t_new)
        t_peaks = r_peaks_original / original_fs
        r_peaks_new = np.round(t_peaks * new_fs).astype(int)
        
        r_peaks_new = r_peaks_new[(r_peaks_new >= 0) & (r_peaks_new < len(new_signal))]
      
        return new_signal, r_peaks_new

def segment_ecg_by_rpeaks(signal, r_peaks, fs, window_heartbeats=7, shift_N=7, subject=None):
    sig_min, sig_max = np.min(signal), np.max(signal)
    N = len(signal)
    segments_list = []
    subjects_list = [] if subject is not None else None
    min_max = (sig_min, sig_max)
    i = 0
    while i + window_heartbeats - 1 < len(r_peaks):
    # if window_heartbeats - 1 < len(r_peaks):
        start_peak = r_peaks[0]
        end_peak = r_peaks[0 + window_heartbeats - 1]
        offset_samples = int(0.15 * fs)
        
        start_idx = max(start_peak - offset_samples, 0)
        end_idx = min(end_peak + offset_samples, N-1)
        segment = signal[start_idx:end_idx+1]
        segments_list.append(segment)
        
       
        if subject is not None:
            subjects_list.append(subject)
        i += shift_N
          
    return segments_list, subjects_list, min_max


def to_fixed_length(signal_list, L):
    new_list = []
    lenth_info=[]
    for segment in signal_list:
        t_original = np.linspace(0, 1, len(segment))
        t_new = np.linspace(0, 1, L)
        f = interp1d(t_original, segment, kind='cubic', fill_value="extrapolate")  
        new_segment = f(t_new)  
        new_list.append(new_segment)
        lenth_info.append((len(segment), L))
    return new_list,lenth_info



def segment_ecg_by_rpeaks(signal1, signal2, r_peaks1, r_peaks2, fs, window_heartbeats=7, shift_N=1, subject=None):
    sig_min1, sig_max1 = np.min(signal1), np.max(signal1)
    signal_norm1 = (signal1 - sig_min1) / (sig_max1 - sig_min1)
    sig_min2, sig_max2 = np.min(signal2), np.max(signal2)
    signal_norm2 = (signal2 - sig_min2) / (sig_max2 - sig_min2)
    N1 = len(signal_norm1)
    N2 = len(signal_norm2)
    segments_list1 = []
    segments_list2 = []
    subjects_list = [] if subject is not None else None
    i = 0
    while i + window_heartbeats - 1 < len(r_peaks1):
        start_peak1 = r_peaks1[i]
        end_peak1 = r_peaks1[i + window_heartbeats - 1]
        offset_samples = int(0.15 * fs)
        
        start_idx1 = max(start_peak1 - offset_samples, 0)
        end_idx1 = min(end_peak1 + offset_samples, N1-1)
        segment1 = signal_norm1[start_idx1:end_idx1+1]
        segments_list1.append(segment1)
        
        i2 = (i + random.randint(1, len(r_peaks2)-1))% (len(r_peaks2) - window_heartbeats + 1)
        start_peak2 = r_peaks2[i2]
        end_peak2 = r_peaks2[i2 + window_heartbeats - 1]
        start_idx2 = max(start_peak2 - offset_samples, 0)
        end_idx2 = min(end_peak2 + offset_samples, N2-1)
        segment2 = signal_norm2[start_idx2:end_idx2+1]
        segments_list2.append(segment2)
       
        if subject is not None:
            subjects_list.append(subject)
        
        i += shift_N
            
    return segments_list1, segments_list2, subjects_list
