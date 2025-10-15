
import os
from tqdm import tqdm 
import pandas as pd
from tools import *
import pickle
import wfdb


def get_RestExer(in_path, sampling_rate, num_heartbeats, shift_N,len_seq,out_path):
    # in_path,sampling_rate,num_heartbeats,shift_N,len_seq,store_aux,out_path
    rest_list,exer_list,sub_list,aux_list = [], [], [], []
    for i in tqdm(range(45)):
        rest_path = os.path.join(in_path,f"{2*i+1}.txt")
        exer_path = os.path.join(in_path,f"{2*i+2}.txt")

        with open(rest_path, 'r', encoding='utf-8') as file:
            line = file.readline()
            rest = np.fromstring(line, sep=' ')
                
        with open(exer_path, 'r', encoding='utf-8') as file:
            line = file.readline()
            exer = np.fromstring(line, sep=' ')

        butterworth_rest = butterworth_lowpass_filter(rest, fs=300, passband=50, stopband=60, 
                              pass_ripple=1.0, stop_atten=2.5)
        butterworth_exer = butterworth_lowpass_filter(exer, fs=300, passband=50, stopband=60, 
                              pass_ripple=1.0, stop_atten=2.5)
        
        resampl_rest, r_peaks_rest = resample_signal_with_rpeaks(butterworth_rest, original_fs=300, new_fs=sampling_rate)
        resampl_exer, r_peaks_exer = resample_signal_with_rpeaks(butterworth_exer, original_fs=300, new_fs=sampling_rate)

        segments_list_rest, segments_list_exer, subjects_list = segment_ecg_by_rpeaks(resampl_rest, resampl_exer, r_peaks_rest, r_peaks_exer, fs=sampling_rate, window_heartbeats=num_heartbeats, shift_N=shift_N, subject=i+1)
        segments_list_rest, len_info_r = to_fixed_length(segments_list_rest, L=len_seq)
        segments_list_exer, len_info_e = to_fixed_length(segments_list_exer, L=len_seq)
        for i in range(len(len_info_r)):
            aux_info = dict()
            aux_info['original_fs'] = 500
            aux_info['new_fs'] = 300
                
            aux_info['len_info_r'] = len_info_r[i]
            aux_info['len_info_e'] = len_info_e[i]
            aux_list.append(aux_info)
            
        rest_list.extend(segments_list_rest)
        exer_list.extend(segments_list_exer)
        sub_list.extend(subjects_list)
    data_pair = {'rest':rest_list, 'exer':exer_list, 'sub':sub_list, 'aux':aux_list} 
    with open(os.path.join(out_path,f'RestExer.pkl'), "wb") as output_file:
        pickle.dump(data_pair, output_file)   